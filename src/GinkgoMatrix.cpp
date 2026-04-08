#ifdef HPGMP_WITH_GINKGO

#include "GinkgoOptData.hpp"
#include "Profiling.hpp"

template<typename local_scalar_t, typename halo_scalar_t>
GinkgoMatrix<local_scalar_t, halo_scalar_t>::GinkgoMatrix(const SparseMatrix<local_scalar_t, halo_scalar_t>& A)
    : ELLMatrix<local_scalar_t, halo_scalar_t>(A)
{

    HPGMP_RANGE_PUSH(__FUNCTION__);

    assert(this->ldi_ == this->ldv_);
    auto gko_exec = create_ginkgo_executor();
    auto ell_mat =
        gko::share(gko_ell_type::create(gko_exec,
                                        gko::dim<2>{static_cast<gko::size_type>(this->local_nrows_),
                                                    static_cast<gko::size_type>(this->local_nrows_)},
                                        std::move(gko::make_array_view(gko_exec,
                                                                       this->ldv_ * this->ell_width_,
                                                                       this->values_)),
                                        std::move(gko::make_array_view(gko_exec,
                                                                       this->ldi_ * this->ell_width_,
                                                                       this->col_idxs_)),
                                        this->ell_width_,
                                        this->ldv_));

    if constexpr (std::is_same_v<gko_mat_type, gko_ell_type>)
    {
        gko_mat_ = ell_mat;
        std::cout << "Using Ginkgo ELL matrix.\n";
    } else if constexpr (std::is_same_v<gko_mat_type, gko_amp_type>)
    {
        std::ifstream amp_config_file("amp_config.txt");
        
        local_scalar_t tol = 1e-8;
        if (amp_config_file.is_open()) {
            amp_config_file >> tol;
            amp_config_file.close();
        }

        auto amp_mat =
            gko::share(gko_amp_type::build().with_tolerance(tol).on(gko_exec)->generate(std::move(ell_mat)));
        gko_mat_ = amp_mat;

        std::cout << "Using Ginkgo AMP matrix with tolerance: " << tol << ".\n";
       
        auto q = gko_mat_->num_precisions;
        for (auto k = 0; k < q; k++)
        {
            const auto* bin = gko_mat_->get_bin_matrix(k);
            const auto* bin_ell = static_cast<const gko_ell_type*>(bin);
            const auto max_nnz_per_row = bin_ell->get_num_stored_elements_per_row();
            std::cout << "Bin " << k  << ": max_nnz_per_row = " << max_nnz_per_row << "\n";
        }
    } else
    {
        throw std::runtime_error("Unsupported gko_mat_type in GinkgoMatrix!");
    }

    HPGMP_RANGE_POP(__FUNCTION__);
}

template<typename local_scalar_t, typename halo_scalar_t, typename vec_scalar_t>
int ginkgo_interior_spmv(const GinkgoMatrix<local_scalar_t, halo_scalar_t>* mat,
                         const Vector<vec_scalar_t>* x, Vector<vec_scalar_t>* y)
{

    HPGMP_RANGE_PUSH(__FUNCTION__);

    using gko_vec_type = gko::matrix::Dense<vec_scalar_t>;
    auto gko_exec      = mat->get_gko_mat()->get_executor();
    auto gko_x =
        gko_vec_type::create_const(gko_exec,
                                   gko::dim<2>{static_cast<gko::size_type>(mat->get_local_num_rows()), 1},
                                   gko::make_const_array_view(gko_exec,
                                                              x->local_length(),
                                                              x->d_values()),
                                   1);
    auto gko_y =
        gko_vec_type::create(gko_exec,
                             gko::dim<2>{static_cast<gko::size_type>(mat->get_local_num_rows()), 1},
                             std::move(gko::make_array_view(gko_exec,
                                                            y->local_length(),
                                                            y->d_values())),
                             1);

    mat->get_gko_mat()->apply(gko_x, gko_y);

    HPGMP_RANGE_POP(__FUNCTION__);

    return 0;
}

template<typename local_scalar_t, typename halo_scalar_t, typename vec_scalar_t>
void ginkgo_spmv(const GinkgoMatrix<local_scalar_t, halo_scalar_t>* mat,
                 const Vector<vec_scalar_t>* x, Vector<vec_scalar_t>* y)
{

    HPGMP_RANGE_PUSH(__FUNCTION__);

    auto dctx = x->get_device_context();

    // On halo stream: pack send buffer and copy to host if needed
    x->update_halos_pack_send_buffer(mat);

    //std::cout << "Using Ginkgo SPMV.\n";
    ginkgo_interior_spmv<local_scalar_t, halo_scalar_t, vec_scalar_t>(mat, x, y);

    // wait for comms to complete
    x->update_halos_send_receive(mat);
    x->update_halos_finalize(mat);

    ell_halo_spmv<local_scalar_t, halo_scalar_t, vec_scalar_t>(mat, x, y);

    dctx->synchronize_halo_stream();
    dctx->synchronize_compute_stream();

    HPGMP_RANGE_POP(__FUNCTION__);
}

template<class SparseMatrix_type, class Vector_type>
int ComputeSPMV_ginkgo(const SparseMatrix_type& A, Vector_type& x, Vector_type& y)
{

    HPGMP_RANGE_PUSH(__FUNCTION__);

    using local_scalar_t = typename SparseMatrix_type::local_scalar_type;
    using halo_scalar_t  = typename SparseMatrix_type::halo_scalar_type;
    auto gko_data        = static_cast<const GinkgoOptData<local_scalar_t, halo_scalar_t>*>(A.optimizationData);
    ginkgo_spmv(gko_data->mat.get(), &x, &y);

    HPGMP_RANGE_POP(__FUNCTION__);

    return 0;
}

// Available template instantiations
template class GinkgoMatrix<double, double>;
template class GinkgoMatrix<float, float>;
template class GinkgoMatrix<double, float>;

template int ginkgo_interior_spmv<double, double, double>(
    const GinkgoMatrix<double, double>*, const Vector<double>*, Vector<double>*);

template int ginkgo_interior_spmv<float, float, float>(
    const GinkgoMatrix<float, float>*, const Vector<float>*, Vector<float>*);

template int ginkgo_interior_spmv<double, float, float>(
    const GinkgoMatrix<double, float>*, const Vector<float>*, Vector<float>*);

template int ginkgo_interior_spmv<double, float, double>(
    const GinkgoMatrix<double, float>*, const Vector<double>*, Vector<double>*);

template int ComputeSPMV_ginkgo< SparseMatrix<double>, Vector<double> >(
    const SparseMatrix<double>&, Vector<double>&, Vector<double>&);

template int ComputeSPMV_ginkgo< SparseMatrix<float>, Vector<float> >(
    const SparseMatrix<float>&, Vector<float>&, Vector<float>&);

template int ComputeSPMV_ginkgo< SparseMatrix<double, float>, Vector<float> >(
    const SparseMatrix<double, float>&, Vector<float>&, Vector<float>&);

template int ComputeSPMV_ginkgo< SparseMatrix<double, float>, Vector<double> >(
    const SparseMatrix<double, float>&, Vector<double>&, Vector<double>&);

#endif
