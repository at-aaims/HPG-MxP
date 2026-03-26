#ifdef HPGMP_WITH_GINKGO
#include "GinkgoInterface.hpp"

std::shared_ptr<gko::Executor> create_ginkgo_executor()
{
#if defined(HPGMP_WITH_HIP) || defined(HPGMP_WITH_CUDA)
    return gko_exec_type::create(0, gko_reference_exec_type::create());
#else
    return gko_exec_type::create();
#endif
}

template<typename mscalar, typename vscalar>
int ginkgo_ell_interior_spmv(const ELLMatrix<mscalar>* mat, const Vector<vscalar>* x, Vector<vscalar>* y)
{
    using gmres        = gko::solver::Gmres<>;
    using bj           = gko::preconditioner::Jacobi<>;
    using gko_vec_type = gko::matrix::Dense<vscalar>;
    using gko_ell_type = gko::matrix::Ell<mscalar, local_int_t>;
    using gko_amp_type = gko::matrix::AMP<mscalar, local_int_t>;
    auto gko_exec      = create_ginkgo_executor();
    auto gko_ell_mat =
        gko::share(gko_ell_type::create_const(gko_exec,
                                              gko::dim<2>{static_cast<gko::size_type>(mat->get_local_num_rows()),
                                                          static_cast<gko::size_type>(mat->get_local_num_cols())},
                                              gko::make_const_array_view(gko_exec,
                                                                         mat->get_ld_values() * mat->get_ell_width(),
                                                                         mat->get_values()),
                                              gko::make_const_array_view(gko_exec,
                                                                         mat->get_ld_indices() * mat->get_ell_width(),
                                                                         mat->get_col_idxs()),
                                              mat->get_ell_width(),
                                              mat->get_ld_values()));
    auto gko_amp_mat = gko_amp_type::build().with_tolerance(1e-1).on(gko_exec)->generate(gko_ell_mat);
    std::cout << "gko_amp_mat->num_precisions:" << gko_amp_mat->num_precisions << "\n";
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

    gko_amp_mat->apply(gko_x, gko_y);

    return 0;
}

template int ginkgo_ell_interior_spmv<double, double>(
    const ELLMatrix<double>*, const Vector<double>*, Vector<double>*);

template int ginkgo_ell_interior_spmv<float, float>(
    const ELLMatrix<float>*, const Vector<float>*, Vector<float>*);

#endif
