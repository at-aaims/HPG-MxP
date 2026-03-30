#ifdef HPGMP_WITH_GINKGO

#include "GinkgoSolver.hpp"

template<typename hiscalar, typename loscalar>
GinkgoSolver<hiscalar, loscalar>::GinkgoSolver(const GinkgoMatrix<hiscalar>* mat)
{
    auto gko_mat        = mat->get_gko_mat();
    auto gko_exec       = gko_mat->get_executor();
    auto color_ptrs     = mat->get_independent_set_offsets();

    // TODO: make the interface take in the data pointer or a const array
    std::vector<local_int_t> color_ptrs_vector(color_ptrs, color_ptrs + mat->get_num_independent_sets()); 
                                                                                                         
    auto solver_factory = solver_type::build()
                              .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
                              .with_color_ptrs(color_ptrs_vector)
                              .on(gko_exec);

    solver_ = gko::share(solver_factory->generate(gko_mat));

    std::cout << "Using Ginkgo (FwdGaussSeidel) solver.\n";
}

// Available template instantiations
template class GinkgoSolver<double, double>;
template class GinkgoSolver<float, float>;

#endif
