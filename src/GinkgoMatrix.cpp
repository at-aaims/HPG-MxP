#ifdef HPGMP_WITH_GINKGO

#include "GinkgoMatrix.hpp"
#include "Profiling.hpp"

template<typename hiscalar, typename loscalar>
GinkgoMatrix<hiscalar, loscalar>::GinkgoMatrix(const SparseMatrix<hiscalar>& A)
    : ELLMatrix<hiscalar, loscalar>(A)
{
    assert(this->ldi_ == this->ldv_);
    auto gko_exec = create_ginkgo_executor();
    auto ell_mat =
        gko::share(gko_ell_type::create(gko_exec,
                                        gko::dim<2>{static_cast<gko::size_type>(this->local_nrows_),
                                                    static_cast<gko::size_type>(this->local_ncols_)},
                                        gko::make_array_view(gko_exec,
                                                             this->ldv_ * this->ell_width_,
                                                             this->values_),
                                        gko::make_array_view(gko_exec,
                                                             this->ldi_ * this->ell_width_,
                                                             this->col_idxs_),
                                        this->ell_width_,
                                        this->ldv_));

    if constexpr (std::is_same_v<gko_mat_type, gko_ell_type>)
    {
        gko_mat_ = ell_mat;
    } else if constexpr (std::is_same_v<gko_mat_type, gko_amp_type>)
    {
        auto amp_mat =
            gko::share(gko_amp_type::build().with_tolerance(1e-1).on(gko_exec)->generate(std::move(ell_mat)));
        gko_mat_ = amp_mat;
        std::cout << "Using Ginkgo AMP matrix.\n";
        std::cout << "amp_mat->num_precisions:" << amp_mat->num_precisions << "\n";
    } else
    {
        throw std::runtime_error("Unsupported gko_mat_type in GinkgoMatrix!");
    }
}

template<typename mscalar, typename vscalar>
int ginkgo_interior_spmv(const GinkgoMatrix<mscalar>* mat, const Vector<vscalar>* x, Vector<vscalar>* y)
{
    using gko_vec_type = gko::matrix::Dense<vscalar>;
    auto gko_exec      = mat->get_gko_mat()->get_executor();
    auto gko_x =
        gko_vec_type::create_const(gko_exec,
                                   gko::dim<2>{static_cast<gko::size_type>(x->local_length()), 1},
                                   gko::make_const_array_view(gko_exec,
                                                              x->local_length(),
                                                              x->d_values()),
                                   1);
    auto gko_y =
        gko_vec_type::create(gko_exec,
                             gko::dim<2>{static_cast<gko::size_type>(y->local_length()), 1},
                             gko::make_array_view(gko_exec,
                                                  y->local_length(),
                                                  y->d_values()),
                             1);

    mat->get_gko_mat()->apply(gko_x, gko_y);

    return 0;
}

template<typename mscalar, typename vscalar>
void ginkgo_spmv(const GinkgoMatrix<mscalar>* mat, const Vector<vscalar>* x, Vector<vscalar>* y)
{
    auto dctx = x->get_device_context();

    // On halo stream: pack send buffer and copy to host if needed
    x->update_halos_pack_send_buffer(mat);

    std::cout << "Using Ginkgo SPMV.\n";
    ginkgo_interior_spmv<mscalar, vscalar>(mat, x, y);

    // wait for comms to complete
    x->update_halos_send_receive(mat);
    x->update_halos_finalize(mat);

    ell_halo_spmv(mat, x, y);

    dctx->synchronize_halo_stream();
    dctx->synchronize_compute_stream();
}

template<class SparseMatrix_type, class Vector_type>
int ComputeSPMV_ginkgo(const SparseMatrix_type& A, Vector_type& x, Vector_type& y)
{

    HPGMP_RANGE_PUSH(__FUNCTION__);

    using scalar_type = typename SparseMatrix_type::scalar_type;
    auto gko_data     = static_cast<const GinkgoOptData<scalar_type, scalar_type>*>(A.optimizationData);
    ginkgo_spmv(gko_data->mat.get(), &x, &y);

    HPGMP_RANGE_POP(__FUNCTION__);

    return 0;
}

// Available template instantiations
template class GinkgoMatrix<double, double>;
template class GinkgoMatrix<float, float>;

template int ginkgo_interior_spmv<double, double>(
    const GinkgoMatrix<double>*, const Vector<double>*, Vector<double>*);

template int ginkgo_interior_spmv<float, float>(
    const GinkgoMatrix<float>*, const Vector<float>*, Vector<float>*);

template int ComputeSPMV_ginkgo< SparseMatrix<double>, Vector<double> >(
    const SparseMatrix<double>&, Vector<double>&, Vector<double>&);

template int ComputeSPMV_ginkgo< SparseMatrix<float>, Vector<float> >(
    const SparseMatrix<float>&, Vector<float>&, Vector<float>&);

#endif
