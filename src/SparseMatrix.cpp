#include "SparseMatrix.hpp"

template <class SparseMatrix_type>
void DeleteMatrix(SparseMatrix_type & A)
{
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

  delete A.optimizationData;

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

  delete [] A.sizes;
  delete [] A.offsets;
  delete [] A.perm;
  return;
}

template void DeleteMatrix(SparseMatrix<double>&);
template void DeleteMatrix(SparseMatrix<float>&);
