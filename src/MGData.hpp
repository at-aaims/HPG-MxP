
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
 @file MGData.hpp

 HPGMP data structure
 */

#ifndef MGDATA_HPP
#define MGDATA_HPP

#include <cassert>
#include "DataTypes.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"

template<class SC>
class MGData {
public:
  typedef Vector<SC> Vector_type;

  MGData(DeviceCtx *const dctx, local_int_t *f2c_operator, Vector<SC> *rvec, Vector<SC> *xvec,
         Vector<SC>* axfvec)
  {
      initialize(dctx, f2c_operator, rvec, xvec, axfvec);
  }

  /** @brief Move-initializer for MG-related vectors.
   * 
   * This class takes ownership of these Vectors and buffer,
   * and deallocates them when it's destroyed.
   */
  void initialize(DeviceCtx *const dctx, local_int_t *f2c_operator, Vector<SC> *rvec, Vector<SC> *xvec,
                  Vector<SC>* axfvec)
  {
      dctx_ = dctx;
      f2cOperator = f2c_operator;
      rc = rvec;
      xc = xvec;
      Axf = axfvec;
  }

  ~MGData()
  {
      delete [] f2cOperator;
      delete Axf;
      delete rc;
      delete xc;
      dctx_->device_free(buffer_R);
      dctx_->device_free(buffer_P);
  }

  DeviceCtx *dctx_;
  int numberOfPresmootherSteps = 1; // Call ComputeSYMGS this many times prior to coarsening
  int numberOfPostsmootherSteps = 1; // Call ComputeSYMGS this many times after coarsening
  local_int_t * f2cOperator; //!< 1D array containing the fine operator local IDs that will be injected into coarse space.
  Vector_type * rc; // coarse grid residual vector
  Vector_type * xc; // coarse grid solution vector
  Vector_type * Axf; // fine grid residual vector
  /*!
   This is for storing optimized data structres created in OptimizeProblem and
   used inside optimized ComputeSPMV().
   */
  void * optimizationData;
  size_t buffer_size_R;
  size_t buffer_size_P;
  void* buffer_R = nullptr;
  void* buffer_P = nullptr;
  #if defined(HPGMP_WITH_CUDA) | defined(HPGMP_WITH_HIP)
  // to store the restrictiion as CRS matrix on device
  int *d_row_ptr;
  int *d_col_idx;
  SC  *d_nzvals;   //!< values of matrix entries
  #if defined(HPGMP_WITH_CUDA)
  cusparseMatDescr_t descrR;
  #elif defined(HPGMP_WITH_HIP)
  rocsparse_spmat_descr descrR;

  // to store transpose
  rocsparse_spmat_descr descrP;
  int *d_tran_row_ptr;
  int *d_tran_col_idx;
  SC  *d_tran_nzvals;   //!< values of matrix entries
  #endif
  #endif
};

//template <class MGData_type>
//inline void DeleteMGData(MGData_type & data) {
//
//  delete [] data.f2cOperator;
//  DeleteVector(*data.Axf);
//  DeleteVector(*data.rc);
//  DeleteVector(*data.xc);
//  delete data.Axf;
//  delete data.rc;
//  delete data.xc;
//}

#endif // MGDATA_HPP

