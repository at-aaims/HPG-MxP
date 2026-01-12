
//@HEADER
// ***************************************************
//
// HPGMP: High Performance Generalized minimal residual
//        - Mixed-Precision
//
// Contact:
// Ichitaro Yamazaki         (iyamaza@sandia.gov)
// Sivasankaran Rajamanickam (srajama@sandia.gov)
// Piotr Luszczek            (luszczek@eecs.utk.edu)
// Jack Dongarra             (dongarra@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file SparseMatrix.hpp

 HPGMP data structures for the sparse matrix
 */

#ifndef HPGMP_SPARSEMATRIX_HPP
#define HPGMP_SPARSEMATRIX_HPP

#include <vector>
#include <cassert>
#include "DataTypes.hpp"
#include "Geometry.hpp"
#include "Vector.hpp"
#include "MGData.hpp"
#include "optimization_base.hpp"

#include <unordered_map>

using GlobalToLocalMap = std::unordered_map< global_int_t, local_int_t >;

#ifndef HPGMP_NO_MPI
#include <mpi.h>
#endif

template<class SC = double>
class SparseMatrix
{
public:
    typedef SC scalar_type;

    void initialize(const Geometry* const g, comm_type c, DeviceCtx* const d)
    {
        dctx = d;
        geom = g;
        comm = c;
    }

    void delete_host_data();

    DeviceCtx* dctx                      = nullptr;
    char* title                          = nullptr; //!< name of the sparse matrix
    const Geometry* geom                 = nullptr; //!< geometry associated with this matrix
    global_int_t totalNumberOfRows       = 0; //!< total number of matrix rows across all processes
    global_int_t totalNumberOfNonzeros   = 0; //!< total number of matrix nonzeros across all processes
    global_int_t totalNumberOfMGNonzeros = 0; //!< total number of matrix nonzeros across all processes, for MG
    local_int_t localNumberOfRows        = 0; //!< number of rows local to this process
    local_int_t localNumberOfColumns     = 0; //!< number of columns local to this process
    local_int_t localNumberOfNonzeros    = 0; //!< number of nonzeros local to this process
    local_int_t localNumberOfMGNonzeros  = 0; //!< number of nonzeros local to this process, for MG
    char* nonzerosInRow                  = nullptr; //!< The number of nonzeros in a row will always be 27 or fewer
    global_int_t** mtxIndG               = nullptr; //!< matrix indices as global values
    local_int_t** mtxIndL                = nullptr; //!< matrix indices as local values
    SC** matrixValues                    = nullptr; //!< values of matrix entries
    SC** matrixDiagonal                  = nullptr; //!< values of matrix diagonal entries
    GlobalToLocalMap globalToLocalMap; //!< global-to-local mapping
    std::vector< global_int_t > localToGlobalMap; //!< local-to-global mapping

    mutable bool isDotProductOptimized = true;
    mutable bool isSpmvOptimized       = true;
    mutable bool isMgOptimized         = true;
    mutable bool isWaxpbyOptimized     = true;
    mutable bool isGemvOptimized       = true;
    /*!
   This is for storing optimized data structres created in OptimizeProblem and
   used inside optimized ComputeSPMV().
   */
    mutable SparseMatrix<SC>* Ac       = nullptr; // Coarse grid matrix
    mutable MGData<SC>* mgData         = nullptr; // Pointer to the coarse level data for this fine matrix
    OptimizationData* optimizationData = nullptr; // pointer that can be used to store implementation-specific data

    // communicator
    comm_type comm;
#ifndef HPGMP_NO_MPI
    local_int_t numberOfExternalValues = 0; //!< number of entries that are external to this process
    int numberOfSendNeighbors          = 0; //!< number of neighboring processes that will be send local data
    local_int_t totalToBeSent          = 0; //!< total number of entries to be sent
    local_int_t* elementsToSend        = nullptr; //!< elements to send to neighboring processes
    int* neighbors                     = nullptr; //!< neighboring processes
    local_int_t* receiveLength         = nullptr; //!< lenghts of messages received from neighboring processes
    local_int_t* sendLength            = nullptr; //!< lenghts of messages sent to neighboring processes
    SC* sendBuffer                     = nullptr; //!< send buffer for non-blocking sends
#if defined(HPGMP_WITH_CUDA) | defined(HPGMP_WITH_HIP)
    local_int_t* d_elementsToSend = nullptr; //!< elements to send to neighboring processes (on GPU)
    SC* d_sendBuffer              = nullptr; //!< send buffer for non-blocking sends (on GPU)
#endif
#endif

#ifdef HPGMP_REFERENCE
#if defined(HPGMP_WITH_CUDA) | defined(HPGMP_WITH_HIP)
#if defined(HPGMP_WITH_CUDA)
    cusparseMatDescr_t descrA;
#elif defined(HPGMP_WITH_HIP)
    rocsparse_spmat_descr descrA;
#endif
    size_t buffer_size_A{};
    void* buffer_A = nullptr;

    // to store the local matrix on device
    int* d_row_ptr = nullptr;
    int* d_col_idx = nullptr;
    SC* d_nzvals   = nullptr; //!< values of matrix entries

    // to store the lower-triangular matrix on device
    local_int_t nnzL{};
#if defined(HPGMP_WITH_CUDA)
    cusparseMatDescr_t descrL;
#if (CUDA_VERSION >= 11000)
    void* buffer_L = nullptr;
    csrsv2Info_t infoL;
#else
    cusparseSolveAnalysisInfo_t infoL;
#endif
#elif defined(HPGMP_WITH_HIP)
    rocsparse_spmat_descr descrL;
    size_t buffer_size_L{};
    void* buffer_L = nullptr;
#endif
    int* d_Lrow_ptr = nullptr;
    int* d_Lcol_idx = nullptr;
    SC* d_Lnzvals   = nullptr; //!< values of matrix entries
    // to store the strictly upper-triangular matrix on device
    local_int_t nnzU{};
#if defined(HPGMP_WITH_CUDA)
    cusparseMatDescr_t descrU;
#elif defined(HPGMP_WITH_HIP)
    rocsparse_spmat_descr descrU;
#endif
    size_t buffer_size_U{};
    void* buffer_U  = nullptr;
    int* d_Urow_ptr = nullptr;
    int* d_Ucol_idx = nullptr;
    SC* d_Unzvals   = nullptr; //!< values of matrix entries

    // workspace vector for reference (vendor library) GS
    mutable Vector<SC> workx; // nrow
#endif

#else // HPGMP_REFERENCE

    const int max_nnz_per_row = 27;

    // Optimizations
    local_int_t* rowHash   = nullptr;
    local_int_t* d_rowHash = nullptr;
    local_int_t* d_mtxIndL = nullptr;
    SC* d_matrixValues     = nullptr;

    // Multicolor GS
    int nblocks{}; //!< Number of independent sets
    int ublocks{}; //!< Number of upper triangular sets
    local_int_t* sizes   = nullptr; //!< Number of rows of each independent set
    local_int_t* offsets = nullptr; //!< Pointer to the first row of each independent set
    local_int_t* perm    = nullptr; //!< Permutation obtained by independent set coloring
#endif

    double time1{}, time2{};
};

/*!
  Copy values from matrix diagonal into user-provided vector.

  @param[in] A the known system matrix.
  @param[inout] diagonal  Vector of diagonal values (must be allocated before call to this function).
 */
template<class SparseMatrix_type, class Vector_type>
inline void CopyMatrixDiagonal(SparseMatrix_type& A, Vector_type& diagonal)
{
    typedef typename SparseMatrix_type::scalar_type scalar_type;
    scalar_type** curDiagA = A.matrixDiagonal;
    scalar_type* dv        = diagonal.values();
    assert(A.localNumberOfRows == diagonal.local_length());
    for (local_int_t i = 0; i < A.localNumberOfRows; ++i)
        dv[i] = *(curDiagA[i]);
    return;
}

/*!
  Replace specified matrix diagonal value.

  @param[inout] A The system matrix.
  @param[in] diagonal  Vector of diagonal values that will replace existing matrix diagonal values.
 */
template<class SparseMatrix_type, class Vector_type>
inline void ReplaceMatrixDiagonal(SparseMatrix_type& A, Vector_type& diagonal)
{
    typedef typename SparseMatrix_type::scalar_type scalar_type;
    scalar_type** curDiagA = A.matrixDiagonal;
    scalar_type* dv        = diagonal.values();
    assert(A.localNumberOfRows == diagonal.local_length());
    for (local_int_t i = 0; i < A.localNumberOfRows; ++i)
        *(curDiagA[i]) = dv[i];
    return;
}

/*!
  Deallocates the members of the data structure of the known system matrix provided they are not 0.

  @param[in] A the known system matrix
 */
template<class SparseMatrix_type>
void DeleteMatrix(SparseMatrix_type& A);

#endif // SPARSEMATRIX_HPP
