#ifdef HPGMP_WITH_GINKGO

#include "GinkgoOptData.hpp"

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
    int ierr = ginkgo_multicolor_gs_interior(interior_smoother, mat, r, x);
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
