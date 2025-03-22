
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

#ifndef SPARSEMATRIX_HPP
#define SPARSEMATRIX_HPP

#include <vector>
#include <cassert>
#include "DataTypes.hpp"
#include "Geometry.hpp"
#include "Vector.hpp"
#include "MGData.hpp"
//#if __cplusplus < 201103L
//// for C++03
//#include <map>
//typedef std::map< global_int_t, local_int_t > GlobalToLocalMap;
//#else
// for C++11 or greater
#include <unordered_map>
using GlobalToLocalMap = std::unordered_map< global_int_t, local_int_t >;
//#endif

#ifndef HPGMP_NO_MPI
 #include <mpi.h>
#endif


template <class SC = double>
class SparseMatrix {
public:
  typedef SC scalar_type;

  void initialize(const Geometry *const g, comm_type c, DeviceCtx *const d)
  {
      dctx = d;
      geom = g;
      comm = c;
  }

  DeviceCtx *dctx = nullptr;
  char  * title = nullptr; //!< name of the sparse matrix
  const Geometry * geom = nullptr; //!< geometry associated with this matrix
  global_int_t totalNumberOfRows = 0; //!< total number of matrix rows across all processes
  global_int_t totalNumberOfNonzeros = 0; //!< total number of matrix nonzeros across all processes
  global_int_t totalNumberOfMGNonzeros = 0; //!< total number of matrix nonzeros across all processes, for MG
  local_int_t localNumberOfRows = 0; //!< number of rows local to this process
  local_int_t localNumberOfColumns = 0;  //!< number of columns local to this process
  local_int_t localNumberOfNonzeros = 0;  //!< number of nonzeros local to this process
  local_int_t localNumberOfMGNonzeros = 0;  //!< number of nonzeros local to this process, for MG
  char  * nonzerosInRow = nullptr;  //!< The number of nonzeros in a row will always be 27 or fewer
  global_int_t ** mtxIndG = nullptr; //!< matrix indices as global values
  local_int_t ** mtxIndL = nullptr; //!< matrix indices as local values
  SC ** matrixValues = nullptr; //!< values of matrix entries
  SC ** matrixDiagonal = nullptr; //!< values of matrix diagonal entries
  GlobalToLocalMap globalToLocalMap; //!< global-to-local mapping
  std::vector< global_int_t > localToGlobalMap; //!< local-to-global mapping

  mutable bool isDotProductOptimized = true;
  mutable bool isSpmvOptimized = true;
  mutable bool isMgOptimized = true;
  mutable bool isWaxpbyOptimized = true;
  mutable bool isGemvOptimized = true;
  /*!
   This is for storing optimized data structres created in OptimizeProblem and
   used inside optimized ComputeSPMV().
   */
  mutable SparseMatrix<SC> * Ac = nullptr;   // Coarse grid matrix
  mutable MGData<SC> * mgData = nullptr; // Pointer to the coarse level data for this fine matrix
  void * optimizationData = nullptr;  // pointer that can be used to store implementation-specific data

  // communicator
  comm_type comm;
#ifndef HPGMP_NO_MPI
  local_int_t numberOfExternalValues = 0; //!< number of entries that are external to this process
  int numberOfSendNeighbors = 0; //!< number of neighboring processes that will be send local data
  local_int_t totalToBeSent = 0; //!< total number of entries to be sent
  local_int_t * elementsToSend = nullptr; //!< elements to send to neighboring processes
  int * neighbors = nullptr; //!< neighboring processes
  local_int_t * receiveLength = nullptr; //!< lenghts of messages received from neighboring processes
  local_int_t * sendLength = nullptr; //!< lenghts of messages sent to neighboring processes
  SC * sendBuffer = nullptr;   //!< send buffer for non-blocking sends
  #if defined(HPGMP_WITH_CUDA) | defined(HPGMP_WITH_HIP)
  local_int_t * d_elementsToSend = nullptr; //!< elements to send to neighboring processes (on GPU)
  SC * d_sendBuffer = nullptr; //!< send buffer for non-blocking sends (on GPU)
  #endif
#endif
#if defined(HPGMP_WITH_CUDA) | defined(HPGMP_WITH_HIP)
  #if defined(HPGMP_WITH_CUDA)
  cusparseMatDescr_t descrA;
  #elif defined(HPGMP_WITH_HIP)
  rocsparse_spmat_descr descrA;
  #endif
  size_t buffer_size_A;
  void* buffer_A;

  // to store the local matrix on device
  int *d_row_ptr;
  int *d_col_idx;
  SC  *d_nzvals;   //!< values of matrix entries

  // to store the lower-triangular matrix on device
  local_int_t nnzL;
  #if defined(HPGMP_WITH_CUDA)
  cusparseMatDescr_t descrL;
  #if (CUDA_VERSION >= 11000)
  void* buffer_L;
  csrsv2Info_t infoL;
  #else
  cusparseSolveAnalysisInfo_t infoL;
  #endif
  #elif defined(HPGMP_WITH_HIP)
  rocsparse_spmat_descr descrL;
  size_t buffer_size_L;
  void* buffer_L;
  #endif
  int *d_Lrow_ptr;
  int *d_Lcol_idx;
  SC  *d_Lnzvals;   //!< values of matrix entries
  // to store the strictly upper-triangular matrix on device
  local_int_t nnzU;
  #if defined(HPGMP_WITH_CUDA)
  cusparseMatDescr_t descrU;
  #elif defined(HPGMP_WITH_HIP)
  rocsparse_spmat_descr descrU;
  #endif
  size_t buffer_size_U;
  void* buffer_U;
  int *d_Urow_ptr;
  int *d_Ucol_idx;
  SC  *d_Unzvals;   //!< values of matrix entries

  // workspace vector
  mutable Vector<SC> workx; // nrow

  // TODO: remove
  //Vector<SC> y; // ncol
#endif

  double time1, time2;
};

#if 0
/*!
  Initializes the known system matrix data structure members to 0.

  @param[in] A the known system matrix
 */
template<class SparseMatrix_type>
inline void InitializeSparseMatrix(SparseMatrix_type & A, Geometry * geom, comm_type comm) {
  A.title = 0;
  A.geom = geom;
  A.totalNumberOfRows = 0;
  A.totalNumberOfNonzeros = 0;
  A.totalNumberOfMGNonzeros = 0;
  A.localNumberOfRows = 0;
  A.localNumberOfColumns = 0;
  A.localNumberOfNonzeros = 0;
  A.localNumberOfMGNonzeros = 0;
  A.nonzerosInRow = 0;
  A.mtxIndG = 0;
  A.mtxIndL = 0;
  A.matrixValues = 0;
  A.matrixDiagonal = 0;

  // Optimization is ON by default. The code that switches it OFF is in the
  // functions that are meant to be optimized.
  A.isDotProductOptimized = true;
  A.isSpmvOptimized       = true;
  A.isMgOptimized         = true;
  A.isWaxpbyOptimized     = true;
  A.isGemvOptimized       = true;

  A.comm = comm;
#ifndef HPGMP_NO_MPI
  A.numberOfExternalValues = 0;
  A.numberOfSendNeighbors = 0;
  A.totalToBeSent = 0;
  A.elementsToSend = 0;
  A.neighbors = 0;
  A.receiveLength = 0;
  A.sendLength = 0;
  A.sendBuffer = 0;
#endif
  A.mgData = 0; // Fine-to-coarse grid transfer initially not defined.
  A.Ac =0;
  return;
}
#endif

/*!
  Copy values from matrix diagonal into user-provided vector.

  @param[in] A the known system matrix.
  @param[inout] diagonal  Vector of diagonal values (must be allocated before call to this function).
 */
template <class SparseMatrix_type, class Vector_type>
inline void CopyMatrixDiagonal(SparseMatrix_type & A, Vector_type & diagonal) {
  typedef typename SparseMatrix_type::scalar_type scalar_type;
  scalar_type ** curDiagA = A.matrixDiagonal;
  scalar_type * dv = diagonal.values();
  assert(A.localNumberOfRows==diagonal.local_length());
  for (local_int_t i=0; i<A.localNumberOfRows; ++i)
      dv[i] = *(curDiagA[i]);
  return;
}
/*!
  Replace specified matrix diagonal value.

  @param[inout] A The system matrix.
  @param[in] diagonal  Vector of diagonal values that will replace existing matrix diagonal values.
 */
template <class SparseMatrix_type, class Vector_type>
inline void ReplaceMatrixDiagonal(SparseMatrix_type & A, Vector_type & diagonal) {
  typedef typename SparseMatrix_type::scalar_type scalar_type;
  scalar_type ** curDiagA = A.matrixDiagonal;
  scalar_type * dv = diagonal.values();
  assert(A.localNumberOfRows==diagonal.local_length());
  for (local_int_t i=0; i<A.localNumberOfRows; ++i)
      *(curDiagA[i]) = dv[i];
  return;
}
/*!
  Deallocates the members of the data structure of the known system matrix provided they are not 0.

  @param[in] A the known system matrix
 */
template <class SparseMatrix_type>
inline void DeleteMatrix(SparseMatrix_type & A) {

#ifndef HPGMP_CONTIGUOUS_ARRAYS
  for (local_int_t i = 0; i< A.localNumberOfRows; ++i) {
    delete [] A.matrixValues[i];
    delete [] A.mtxIndG[i];
    delete [] A.mtxIndL[i];
  }
#else
  delete [] A.matrixValues[0];
  delete [] A.mtxIndG[0];
  delete [] A.mtxIndL[0];
#endif
  if (A.title)                 delete [] A.title;
  if (A.nonzerosInRow)         delete [] A.nonzerosInRow;
  if (A.mtxIndG)               delete [] A.mtxIndG;
  if (A.mtxIndL)               delete [] A.mtxIndL;
  if (A.matrixValues)          delete [] A.matrixValues;
  if (A.matrixDiagonal)        delete [] A.matrixDiagonal;

#ifndef HPGMP_NO_MPI
  if (A.elementsToSend)        delete [] A.elementsToSend;
  if (A.neighbors)             delete [] A.neighbors;
  if (A.receiveLength)         delete [] A.receiveLength;
  if (A.sendLength)            delete [] A.sendLength;
  if (A.sendBuffer)            delete [] A.sendBuffer;
#endif

  /*if (A.geom!=0) {
    DeleteGeometry(*A.geom);
    delete A.geom;
    A.geom = 0;
  }*/
  if (A.Ac!=0) {
    // Delete coarse matrix
    DeleteMatrix(*A.Ac);
    delete A.Ac; 
    A.Ac = 0;
  }
  if (A.mgData!=0) {
    // Delete MG data
    //DeleteMGData(*A.mgData);
    delete A.mgData;
    A.mgData = nullptr;
  }

  //DeleteVector (A.workx);
  //DeleteVector (A.y);

#ifdef HPGMP_WITH_CUDA
  cudaFree (A.d_row_ptr);
  cudaFree (A.d_col_idx);
  cudaFree (A.d_nzvals);
  cudaFree (A.d_sendBuffer);
  cudaFree (A.d_elementsToSend);

  cudaFree (A.d_Lrow_ptr);
  cudaFree (A.d_Lcol_idx);
  cudaFree (A.d_Lnzvals);

  cudaFree (A.d_Urow_ptr);
  cudaFree (A.d_Ucol_idx);
  cudaFree (A.d_Unzvals);

  cusparseDestroyMatDescr(A.descrA);
  cusparseDestroyMatDescr(A.descrL);
  cusparseDestroyMatDescr(A.descrU);
  #if (CUDA_VERSION >= 11000)
  cusparseDestroyCsrsv2Info(A.infoL);
  #else
  cusparseDestroySolveAnalysisInfo(A.infoL);
  #endif
#elif defined(HPGMP_WITH_HIP)
  hipFree (A.d_row_ptr);
  hipFree (A.d_col_idx);
  hipFree (A.d_nzvals);
  hipFree (A.d_sendBuffer);
  hipFree (A.d_elementsToSend);

  hipFree (A.d_Lrow_ptr);
  hipFree (A.d_Lcol_idx);
  hipFree (A.d_Lnzvals);

  hipFree (A.d_Urow_ptr);
  hipFree (A.d_Ucol_idx);
  hipFree (A.d_Unzvals);

  rocsparse_destroy_spmat_descr(A.descrA);
  rocsparse_destroy_spmat_descr(A.descrL);
  rocsparse_destroy_spmat_descr(A.descrU);
#endif
  return;
}

#endif // SPARSEMATRIX_HPP
