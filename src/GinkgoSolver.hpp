#ifndef HPGMP_GINKGO_SOLVER_HPP
#define HPGMP_GINKGO_SOLVER_HPP

#include "GinkgoInterface.hpp"
#include "GinkgoMatrix.hpp"

template<typename hiscalar, typename loscalar = hiscalar>
class GinkgoSolver
{
public:
    using scalar_type = hiscalar;
    using solver_type = gko::solver::FwdGaussSeidel<scalar_type, local_int_t>;

    GinkgoSolver(const GinkgoMatrix<scalar_type>* mat);

    ~GinkgoSolver()
    { }

    auto get_solver() const { return solver_; }

protected:
    std::shared_ptr<solver_type> solver_;
};

template<typename mscalar, typename vscalar>
int ginkgo_multicolor_gs(const GinkgoSolver<mscalar>* solver, const Vector<vscalar>* r,
                         Vector<vscalar>* x);

#endif
