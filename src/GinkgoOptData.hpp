#ifndef HPGMP_GINKGO_OPT_DATA_HPP
#define HPGMP_GINKGO_OPT_DATA_HPP

#include "GinkgoMatrix.hpp"
#include "GinkgoSolver.hpp"

template<typename local_scalar_t, typename halo_scalar_t>
struct GinkgoOptData : public EllOptData<local_scalar_t, halo_scalar_t>
{
    using matrix_type = GinkgoMatrix<local_scalar_t, halo_scalar_t>;
    using solver_type = GinkgoSolver<local_scalar_t, halo_scalar_t>;
    std::shared_ptr<matrix_type> mat;
    std::shared_ptr<solver_type> solver;
};

#endif
