#ifndef HPGMP_ELL_MATRIX
#define HPGMP_ELL_MATRIX

#include <memory>

#include "matrix_base.hpp"
#include "SparseMatrix.hpp"
#include "optimization_base.hpp"

template<typename hiscalar, typename loscalar = hiscalar>
class ELLMatrix : public DistMatrixBase
{
public:
    using scalar_type = hiscalar;

    ELLMatrix(comm_type comm, DeviceCtx* const dctx, const Geometry* const geom)
        : DistMatrixBase(comm, dctx, geom)
    { }

    /**
     * Initialize this matrix with a conversion of a HPGMP SparseMatrix.
     */
    ELLMatrix(const SparseMatrix<hiscalar>& A);

    ~ELLMatrix();

    local_int_t get_ell_width() const { return ell_width_; }
    const hiscalar* get_values() const { return values_; }
    const local_int_t* get_col_idxs() const { return col_idxs_; }

    const local_int_t* get_halo_col_idxs() const { return halo_col_idxs_; }
    const hiscalar* get_halo_values() const { return halo_values_; }

    int get_ld_values() const { return ldv_; }
    int get_ld_indices() const { return ldi_; }
    int get_halo_ld_values() const { return halo_ldv_; }
    int get_halo_ld_indices() const { return halo_ldi_; }

    /** @brief Permute rows of the matrix based on a permutation vector.
     *
     * @param perm  Permutation indices s.t. in new matrix, row perm[i] was row i in the old one.
     */
    void permute_rows(const local_int_t* perm);

    const hiscalar* get_inverse_diagonal() const { return inv_diag_; }
    const local_int_t* get_diagonal_indices() const { return diag_idxs_; }

    /** @brief Compute and store indices of diagonal entries in every row and
     *         the inverses of the diagonal entries.
     */
    void extract_diagonal();

protected:
    static constexpr int pad_mult_v = padding_multiple<hiscalar>::value;
    static constexpr int pad_mult_i = padding_multiple<local_int_t>::value;

    /** @brief Leading dimension of values array.
     *
     * Distance (in no. of scalars) between the start of consecutive columns.
     * Minimum is the number of local rows.
     */
    int ldv_{};
    /// Leading dimension of col_idxs_ array.
    int ldi_{};
    int halo_ldv_{};
    int halo_ldi_{};

    /// Max nnz per row in the ELL matrix.
    local_int_t ell_width_{};
    /// Local column indices.
    local_int_t* col_idxs_ = nullptr;
    /// Nonzero values corresponding to column indices in col_idxs_.
    hiscalar* values_ = nullptr;

    local_int_t* halo_col_idxs_ = nullptr;
    hiscalar* halo_values_      = nullptr;

    local_int_t* diag_idxs_ = nullptr;
    hiscalar* inv_diag_     = nullptr;

    void convert_from_csr(const SparseMatrix<hiscalar>& A);
};

//template <typename mscalar, typename vscalar>
//void ell_spmv(const ELLMatrix<mscalar>* mat, const Vector<vscalar> *x, Vector<vscalar>* y);

template<typename mscalar, typename vscalar>
void ell_halo_spmv(const ELLMatrix<mscalar>* mat, const Vector<vscalar>* x, Vector<vscalar>* y);

template<typename mscalar, typename vscalar>
void ell_interior_spmv(const ELLMatrix<mscalar>* mat, const Vector<vscalar>* x, Vector<vscalar>* y);

template<typename hiscalar, typename loscalar = hiscalar>
struct EllOptData : public OptimizationData
{
    std::shared_ptr<ELLMatrix<hiscalar, loscalar>> mat;
};

template<class SparseMatrix_type, class Vector_type>
int ComputeSPMV_ell(const SparseMatrix_type& A, Vector_type& x, Vector_type& y);

#endif
