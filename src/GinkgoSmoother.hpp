#ifndef HPGMP_GINKGO_SOLVER_HPP
#define HPGMP_GINKGO_SOLVER_HPP

#include "GinkgoInterface.hpp"
#include "GinkgoMatrix.hpp"

template<typename local_scalar_t, typename halo_scalar_t>
class GinkgoSmoother
{
public:
    using scalar_type = local_scalar_t;
    using solver_type = gko::solver::FwdGaussSeidel<local_scalar_t, local_int_t>;

    GinkgoSmoother(const GinkgoMatrix<local_scalar_t, halo_scalar_t>* mat);

    ~GinkgoSmoother()
    { }

    auto get_solver() const { return solver_; }

protected:
    std::shared_ptr<solver_type> solver_;
};

template<typename local_scalar_t, typename halo_scalar_t, typename vec_scalar_t>
int ginkgo_multicolor_gs_interior(const GinkgoSmoother<local_scalar_t, halo_scalar_t>* interior_solver,
                                  const GinkgoMatrix<local_scalar_t, halo_scalar_t>* mat,
                                  const Vector<vec_scalar_t>* r, Vector<vec_scalar_t>* x);

template<typename local_scalar_t, typename halo_scalar_t, typename vec_scalar_t>
int ginkgo_multicolor_gs(const GinkgoSmoother<local_scalar_t, halo_scalar_t>* interior_solver,
                         const GinkgoMatrix<local_scalar_t, halo_scalar_t>* mat,
                         const Vector<vec_scalar_t>* r, Vector<vec_scalar_t>* x);

#endif
