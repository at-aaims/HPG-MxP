#ifndef HPGMP_GINKGO_MATRIX_HPP
#define HPGMP_GINKGO_MATRIX_HPP

#include "ell_matrix.hpp"
#include "GinkgoInterface.hpp"

/**
 * Ginkgo matrix selection depending on the input scalar type
 * Note we only use Ginkgo for the local part of the matrix; not the non-local (halo) part.
 * Ell<double> if input_scalar_type is double
 * AMP<float> if input_scalar_type is float
 * TODO: Figure out the instantiation of AMP<double> when performing the
 * mixed-precision benchmark
 */
template<typename input_scalar_type>
struct GinkgoMatrixSelection
{ };

template<>
struct GinkgoMatrixSelection<float>
{
    using scalar_type = float;
    using value       = gko::matrix::AMP<scalar_type, local_int_t>;
};

template<>
struct GinkgoMatrixSelection<double>
{
    using scalar_type = double;
    using value       = gko::matrix::Ell<scalar_type, local_int_t>;
};

template<typename hiscalar, typename loscalar = hiscalar>
class GinkgoMatrix : public ELLMatrix<hiscalar, loscalar>
{
public:
    using scalar_type  = typename GinkgoMatrixSelection<hiscalar>::scalar_type;
    using gko_mat_type = typename GinkgoMatrixSelection<hiscalar>::value;
    using gko_ell_type = gko::matrix::Ell<scalar_type, local_int_t>;
    using gko_amp_type = gko::matrix::AMP<scalar_type, local_int_t>;

    GinkgoMatrix(comm_type comm, DeviceCtx* const dctx, const Geometry* const geom) = delete;

    /**
     * Initialize this matrix with a conversion of a HPGMP SparseMatrix, with
     * ELLMatrix as an intermediary step.
     */
    GinkgoMatrix(const SparseMatrix<hiscalar>& A);

    ~GinkgoMatrix()
    { }

    auto get_gko_mat() const { return gko_mat_; }

protected:
    std::shared_ptr<gko_mat_type> gko_mat_;
};

template<typename hiscalar, typename loscalar = hiscalar>
struct GinkgoOptData : public EllOptData<hiscalar, loscalar>
{
    using matrix_type = GinkgoMatrix<hiscalar, loscalar>;
    std::shared_ptr<matrix_type> mat;
};

template<typename mscalar, typename vscalar>
int ginkgo_interior_spmv(const GinkgoMatrix<mscalar>* mat, const Vector<vscalar>* x, Vector<vscalar>* y);

template<class SparseMatrix_type, class Vector_type>
int ComputeSPMV_ginkgo(const SparseMatrix_type& A, Vector_type& x, Vector_type& y);

#endif
