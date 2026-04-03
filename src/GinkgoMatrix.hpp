#ifndef HPGMP_GINKGO_MATRIX_HPP
#define HPGMP_GINKGO_MATRIX_HPP

#include "ell_matrix.hpp"
#include "GinkgoInterface.hpp"

/**
 * Ginkgo matrix selection depending on the input scalar types
 * Note we only use Ginkgo for the local part of the matrix; not the non-local (halo) part.
 * Ell<> in uniform precision
 * AMP<double> in mixed precision (float inner)
 */
template<typename local_scalar_t, typename halo_scalar_t>
struct GinkgoMatrixSelection
{ };

template<>
struct GinkgoMatrixSelection<double, float>
{
    using value = gko::matrix::AMP<double, local_int_t>;
};

template<>
struct GinkgoMatrixSelection<float, float>
{
    using value = gko::matrix::Ell<float, local_int_t>;
};

template<>
struct GinkgoMatrixSelection<double, double>
{
    using value = gko::matrix::Ell<double, local_int_t>;
};

template<typename local_scalar_t, typename halo_scalar_t>
class GinkgoMatrix : public ELLMatrix<local_scalar_t, halo_scalar_t>
{
public:
    using scalar_type  = local_scalar_t;
    using gko_mat_type = typename GinkgoMatrixSelection<local_scalar_t, halo_scalar_t>::value;
    using gko_ell_type = gko::matrix::Ell<local_scalar_t, local_int_t>;
    using gko_amp_type = gko::matrix::AMP<local_scalar_t, local_int_t>;

    GinkgoMatrix(comm_type comm, DeviceCtx* const dctx, const Geometry* const geom) = delete;

    /**
     * Initialize this matrix with a conversion of a HPGMP SparseMatrix, with
     * ELLMatrix as an intermediary step.
     */
    GinkgoMatrix(const SparseMatrix<local_scalar_t, halo_scalar_t>& A);

    ~GinkgoMatrix()
    { }

    std::shared_ptr<const gko_mat_type> get_gko_mat() const { return gko_mat_; }

protected:
    std::shared_ptr<gko_mat_type> gko_mat_;
};

template<typename local_scalar_t, typename halo_scalar_t, typename vec_scalar_t>
int ginkgo_interior_spmv(const GinkgoMatrix<local_scalar_t, halo_scalar_t>* mat,
                         const Vector<vec_scalar_t>* x, Vector<vec_scalar_t>* y);

template<class SparseMatrix_type, class Vector_type>
int ComputeSPMV_ginkgo(const SparseMatrix_type& A, Vector_type& x, Vector_type& y);

#endif
