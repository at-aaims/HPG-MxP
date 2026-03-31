#ifdef HPGMP_WITH_GINKGO

#include "GinkgoSolver.hpp"

template<typename hiscalar, typename loscalar>
GinkgoSolver<hiscalar, loscalar>::GinkgoSolver(const GinkgoMatrix<hiscalar>* mat)
{
    auto gko_mat    = mat->get_gko_mat();
    auto gko_exec   = gko_mat->get_executor();
    auto color_ptrs = mat->get_independent_set_offsets();

    // TODO: make the interface take in the data pointer or a const array
    std::vector<local_int_t> color_ptrs_vector(color_ptrs, color_ptrs + mat->get_num_independent_sets());

    auto solver_factory = solver_type::build()
                              .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
                              .with_color_ptrs(color_ptrs_vector)
                              .on(gko_exec);

    solver_ = gko::share(solver_factory->generate(gko_mat));

    std::cout << "Using Ginkgo (FwdGaussSeidel) solver.\n";
}

template<typename mscalar, typename vscalar>
int ginkgo_multicolor_gs_interior(const GinkgoSolver<mscalar>* interior_solver, const GinkgoMatrix<mscalar>* mat,
                                  const Vector<vscalar>* r, Vector<vscalar>* x)
{
    using gko_vec_type = gko::matrix::Dense<vscalar>;
    auto gko_solver    = interior_solver->get_solver();
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
                             gko::make_array_view(gko_exec,
                                                  x->local_length(),
                                                  x->d_values()),
                             1);

    gko_solver->apply(gko_r, gko_x);

    return 0;
}

template<typename mscalar, typename vscalar>
int ginkgo_multicolor_gs(const GinkgoSolver<mscalar>* interior_solver, const GinkgoMatrix<mscalar>* mat,
                         const Vector<vscalar>* r, Vector<vscalar>* x)
{
    int ierr = ginkgo_multicolor_gs_interior(interior_solver, mat, r, x);
    return 0;
}

// Available template instantiations
template class GinkgoSolver<double, double>;
template class GinkgoSolver<float, float>;

template int ginkgo_multicolor_gs_interior(const GinkgoSolver<double>* interior_solver, const GinkgoMatrix<double>* mat,
                                           const Vector<double>* r, Vector<double>* x);
template int ginkgo_multicolor_gs_interior(const GinkgoSolver<float>* interior_solver, const GinkgoMatrix<float>* mat,
                                           const Vector<float>* r, Vector<float>* x);

template int ginkgo_multicolor_gs(const GinkgoSolver<double>* interior_solver, const GinkgoMatrix<double>* mat,
                                  const Vector<double>* r, Vector<double>* x);
template int ginkgo_multicolor_gs(const GinkgoSolver<float>* interior_solver, const GinkgoMatrix<float>* mat,
                                  const Vector<float>* r, Vector<float>* x);
#endif
