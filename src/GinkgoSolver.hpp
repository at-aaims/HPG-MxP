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

template<typename mat_scalar_type, typename vec_scalar_type>
int ginkgo_multicolor_gs_interior(const GinkgoSolver<mat_scalar_type, mat_scalar_type>* interior_solver,
                                  const GinkgoMatrix<mat_scalar_type, mat_scalar_type>* mat,
                                  const Vector<vec_scalar_type>* r, Vector<vec_scalar_type>* x);

template<typename mat_scalar_type, typename vec_scalar_type>
int ginkgo_multicolor_gs(const GinkgoSolver<mat_scalar_type, mat_scalar_type>* interior_solver,
                         const GinkgoMatrix<mat_scalar_type, mat_scalar_type>* mat,
                         const Vector<vec_scalar_type>* r, Vector<vec_scalar_type>* x);

#endif
