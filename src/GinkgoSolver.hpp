#ifndef HPGMP_GINKGO_SOLVER_HPP
#define HPGMP_GINKGO_SOLVER_HPP

#include "GinkgoInterface.hpp"
#include "GinkgoMatrix.hpp"

template<typename hiscalar, typename loscalar>
class GinkgoSolver
{
public:
    using scalar_type = hiscalar;
    using solver_type = gko::solver::FwdGaussSeidel<scalar_type, local_int_t>;

    GinkgoSolver(const GinkgoMatrix<scalar_type, scalar_type>* mat);

    ~GinkgoSolver()
    { }

    auto get_solver() const { return solver_; }

protected:
    std::shared_ptr<solver_type> solver_;
};

template<typename mscalar, typename vscalar>
int ginkgo_multicolor_gs_interior(const GinkgoSolver<mscalar, mscalar>* interior_solver,
                                  const GinkgoMatrix<mscalar, mscalar>* mat,
                                  const Vector<vscalar>* r, Vector<vscalar>* x);

template<typename mscalar, typename vscalar>
int ginkgo_multicolor_gs(const GinkgoSolver<mscalar, mscalar>* interior_solver,
                         const GinkgoMatrix<mscalar, mscalar>* mat,
                         const Vector<vscalar>* r, Vector<vscalar>* x);

#endif
