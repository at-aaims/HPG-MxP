#ifndef HPGMP_GINKGO_SMOOTHER_HPP
#define HPGMP_GINKGO_SMOOTHER_HPP

#include "GinkgoInterface.hpp"
#include "GinkgoMatrix.hpp"

template<typename local_scalar_t, typename halo_scalar_t>
class GinkgoSmoother
{
public:
    using scalar_type   = local_scalar_t;
    using smoother_type = gko::solver::FwdGaussSeidel<local_scalar_t, local_int_t>;

    GinkgoSmoother(const GinkgoMatrix<local_scalar_t, halo_scalar_t>* mat);

    ~GinkgoSmoother()
    { }

    std::shared_ptr<const smoother_type> get_smoother() const { return smoother_; }

protected:
    std::shared_ptr<smoother_type> smoother_;
};

template<typename local_scalar_t, typename halo_scalar_t, typename vec_scalar_t>
int ginkgo_multicolor_gs_interior(const GinkgoSmoother<local_scalar_t, halo_scalar_t>* interior_smoother,
                                  const GinkgoMatrix<local_scalar_t, halo_scalar_t>* mat,
                                  const Vector<vec_scalar_t>* r, Vector<vec_scalar_t>* x);

template<typename local_scalar_t, typename halo_scalar_t, typename vec_scalar_t>
int ginkgo_multicolor_gs(const GinkgoSmoother<local_scalar_t, halo_scalar_t>* interior_smoother,
                         const GinkgoMatrix<local_scalar_t, halo_scalar_t>* mat,
                         const Vector<vec_scalar_t>* r, Vector<vec_scalar_t>* x);

#endif
