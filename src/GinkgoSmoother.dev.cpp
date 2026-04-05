#ifdef HPGMP_WITH_GINKGO

#include "GinkgoOptData.hpp"

#include "kernel_helpers.hpp.inc"

#define LAUNCH_FGS_HALO(blocksize, width)                                              \
    {                                                                                  \
        dim3 blocks((mat->get_num_halo_rows() - 1) / blocksize + 1);                   \
        dim3 threads(blocksize);                                                       \
                                                                                       \
        kernel_fgs_halo<blocksize, width, local_scalar_t, halo_scalar_t, vec_scalar_t> \
            <<<blocks,                                                                 \
               threads,                                                                \
               0,                                                                      \
               stream_interior>>>(                                                     \
                mat->get_num_halo_rows(),                                              \
                mat->get_local_num_cols(),                                             \
                mat->get_independent_set_sizes()[i],                                   \
                mat->get_halo_ld_indices(), mat->get_halo_ld_values(),                 \
                mat->get_halo_row_indices(),                                           \
                mat->get_halo_col_idxs(),                                              \
                mat->get_halo_values(),                                                \
                mat->get_inverse_diagonal(),                                           \
                mat->get_reordering_permutation(),                                     \
                r->d_values(),                                                         \
                x->d_values());                                                        \
    }

template<unsigned int BLOCKSIZE, unsigned int WIDTH,
         typename local_scalar_t, typename halo_scalar_t, typename vec_scalar_t>
__launch_bounds__(BLOCKSIZE)
    __global__ void kernel_fgs_halo(const local_int_t m,
                                      const local_int_t n,
                                      const local_int_t block_nrow,
                                      const int ldi, const int ldv,
                                      const local_int_t* halo_row_ind,
                                      const local_int_t* halo_col_ind,
                                      const halo_scalar_t* halo_val,
                                      const local_scalar_t* inv_diag,
                                      const local_int_t* perm,
                                      const vec_scalar_t* x,
                                      vec_scalar_t* y)
{
    const local_int_t row = blockIdx.x * BLOCKSIZE + threadIdx.x;
    if (row >= m) {
        return;
    }

    const local_int_t halo_idx = __ldcg(halo_row_ind + row);
    const local_int_t perm_idx = perm[halo_idx];
    if (perm_idx >= block_nrow) {
        return;
    }

    vec_scalar_t sum = 0.0;

#pragma unroll
    for (local_int_t p = 0; p < WIDTH; ++p)
    {
        const local_int_t col = __ldcg(halo_col_ind + row + p * ldi);

        if (col >= 0 && col < n) {
            sum = fma(-static_cast<vec_scalar_t>(__ldcg(halo_val + row + p * ldv)),
                      y[col], sum);
        }
    }
    y[perm_idx] = fma(sum, static_cast<vec_scalar_t>(inv_diag[halo_idx]), y[perm_idx]);
}

template<typename local_scalar_t, typename halo_scalar_t>
GinkgoSmoother<local_scalar_t, halo_scalar_t>::GinkgoSmoother(const GinkgoMatrix<local_scalar_t, halo_scalar_t>* mat)
{
    auto gko_mat    = mat->get_gko_mat();
    auto gko_exec   = gko_mat->get_executor();
    auto color_ptrs = mat->get_independent_set_offsets();

    // TODO: make the interface take in the data pointer or a const array
    std::vector<local_int_t> color_ptrs_vector(color_ptrs, color_ptrs + mat->get_num_independent_sets() + 1);

    auto smoother_factory = smoother_type::build()
                                .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
                                .with_color_ptrs(color_ptrs_vector)
                                .on(gko_exec);

    smoother_ = gko::share(smoother_factory->generate(gko_mat));

    //std::cout << "Using Ginkgo (FwdGaussSeidel) smoother.\n";
}

template<typename local_scalar_t, typename halo_scalar_t, typename vec_scalar_t>
int ginkgo_multicolor_gs_interior(const GinkgoSmoother<local_scalar_t, halo_scalar_t>* interior_smoother,
                                  const GinkgoMatrix<local_scalar_t, halo_scalar_t>* mat,
                                  const Vector<vec_scalar_t>* r, Vector<vec_scalar_t>* x)
{
    using gko_vec_type = gko::matrix::Dense<vec_scalar_t>;
    auto gko_smoother  = interior_smoother->get_smoother();
    auto gko_exec      = mat->get_gko_mat()->get_executor();
    auto gko_r =
        gko_vec_type::create_const(gko_exec,
                                   gko::dim<2>{static_cast<gko::size_type>(r->local_length()), 1},
                                   gko::make_const_array_view(gko_exec,
                                                              r->local_length(),
                                                              r->d_values()),
                                   1);
    auto gko_x =
        gko_vec_type::create(gko_exec,
                             gko::dim<2>{static_cast<gko::size_type>(x->local_length()), 1},
                             std::move(gko::make_array_view(gko_exec,
                                                            x->local_length(),
                                                            x->d_values())),
                             1);

    gko_smoother->apply(gko_r, gko_x);

    return 0;
}

template<typename local_scalar_t, typename halo_scalar_t, typename vec_scalar_t>
int ginkgo_multicolor_gs(const GinkgoSmoother<local_scalar_t, halo_scalar_t>* interior_smoother,
                         const GinkgoMatrix<local_scalar_t, halo_scalar_t>* mat,
                         const Vector<vec_scalar_t>* r, Vector<vec_scalar_t>* x)
{
    assert(x->local_length() == mat->get_local_num_cols());
    auto dctx            = mat->get_device_context();
    auto stream_interior = dctx->get_compute_stream();

    local_int_t i = 0;

#ifndef HPGMP_NO_MPI
    if (mat->get_geometry()->size > 1)
    {
        x->update_halos_pack_send_buffer(mat);
    }
#endif

    ginkgo_multicolor_gs_interior(interior_smoother, mat, r, x); // all independent row blocks
                                                                 
#ifndef HPGMP_NO_MPI
    if (mat->get_geometry()->size > 1)
    {
        x->update_halos_send_receive(mat);
        x->update_halos_finalize(mat);
        for (local_int_t i; i < mat->get_num_independent_sets(); ++i) // all colors
        {
            if (mat->get_ell_width() == 27) {
                LAUNCH_FGS_HALO(256, 27);
            }
        }
    }
#endif

    dctx->synchronize_compute_stream();
    dctx->synchronize_halo_stream();

    return 0;
}

// Available template instantiations
template class GinkgoSmoother<double, double>;
template class GinkgoSmoother<float, float>;
template class GinkgoSmoother<double, float>;

template int ginkgo_multicolor_gs_interior(const GinkgoSmoother<double, double>* interior_smoother, const GinkgoMatrix<double, double>* mat,
                                           const Vector<double>* r, Vector<double>* x);
template int ginkgo_multicolor_gs_interior(const GinkgoSmoother<float, float>* interior_smoother, const GinkgoMatrix<float, float>* mat,
                                           const Vector<float>* r, Vector<float>* x);
template int ginkgo_multicolor_gs_interior(const GinkgoSmoother<double, float>* interior_smoother, const GinkgoMatrix<double, float>* mat,
                                           const Vector<float>* r, Vector<float>* x);
template int ginkgo_multicolor_gs_interior(const GinkgoSmoother<double, float>* interior_smoother, const GinkgoMatrix<double, float>* mat,
                                           const Vector<double>* r, Vector<double>* x);

template int ginkgo_multicolor_gs(const GinkgoSmoother<double, double>* interior_smoother, const GinkgoMatrix<double, double>* mat,
                                  const Vector<double>* r, Vector<double>* x);
template int ginkgo_multicolor_gs(const GinkgoSmoother<float, float>* interior_smoother, const GinkgoMatrix<float, float>* mat,
                                  const Vector<float>* r, Vector<float>* x);
template int ginkgo_multicolor_gs(const GinkgoSmoother<double, float>* interior_smoother, const GinkgoMatrix<double, float>* mat,
                                  const Vector<float>* r, Vector<float>* x);
template int ginkgo_multicolor_gs(const GinkgoSmoother<double, float>* interior_smoother, const GinkgoMatrix<double, float>* mat,
                                  const Vector<double>* r, Vector<double>* x);
#endif
