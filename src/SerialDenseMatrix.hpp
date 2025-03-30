
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
 @file Vector.hpp

 HPGMP data structures for dense vectors
 */

#ifndef SERIAL_DENSE_MATRIX_HPP
#define SERIAL_DENSE_MATRIX_HPP

#include <fstream>
#include <cassert>
#include <cstdlib>

#include "DataTypes.hpp"
#include "device_ctx.hpp"
#include "exceptions.hpp"

/** @brief Dense matrix mostly on the host.
 *
 * Stored column-major.
 */
template<class SC>
class SerialDenseMatrix {
public:
  typedef SC scalar_type;

  SerialDenseMatrix(const local_int_t m, const local_int_t n, DeviceCtx *const dctx)
      : m_{m}, n_{n}, values_{new scalar_type[m*n]},
#ifdef HPGMP_WITH_ACCELERATION
      d_values_{(scalar_type*)dctx->device_alloc(m_*n_*sizeof(scalar_type))},
#endif
      dctx_{dctx}
  { }

  ~SerialDenseMatrix()
  {
      delete [] values_;
#ifdef HPGMP_WITH_ACCELERATION
      dctx_->device_free(d_values_);
#endif
      m_ = n_ = 0;
  }

  local_int_t n_rows() const { return m_; }
  local_int_t n_cols() const { return n_; }

  void reshape(const local_int_t m, const local_int_t n) {
      m_ = m;
      n_ = n;
  }

  void fill_zero() {
      const scalar_type zero (0.0);

      for (int i=0; i<m_*n_; ++i) 
        values_[i] = zero;
  }

  void add_value(const local_int_t i, const local_int_t j, const scalar_type value) {
      assert(i>=0 && i < m_);
      assert(j>=0 && j < n_);
      values_[i + j*m_] += value;
  }

  void set_value(const local_int_t i, const local_int_t j, const scalar_type value) {
      assert(i>=0 && i < m_);
      assert(j>=0 && j < n_);
      values_[i + j*m_] = value;
  }

  scalar_type get_value(const local_int_t i, const local_int_t j) const
  {
      assert(i>=0 && i < m_);
      assert(j>=0 && j < n_);
      return values_[i + j*m_];
  }

  const scalar_type* values() const {
      return values_;
  }

  scalar_type* values() {
      return values_;
  }

  const scalar_type* d_values() const {
      return d_values_;
  }

  scalar_type* d_values() {
      return d_values_;
  }

  /// Updates the device data of the matrix from the host buffer's data.
  void update_device_data() const {
#ifdef HPGMP_WITH_CUDA
      if(cudaSuccess != cudaMemcpy(d_values_, values_, m_*n_*sizeof(scalar_type),
                                   cudaMemcpyHostToDevice)) {
          throw HostDeviceCopyFailedError("Failed to update device data!");
      }
#elif defined HPGMP_WITH_HIP
      if(hipSuccess != hipMemcpy(d_values_, values_, m_*n_*sizeof(scalar_type),
                                 hipMemcpyHostToDevice)) {
          throw HostDeviceCopyFailedError("Failed to update device data!");
      }
#endif
  }

  /// Updates the device data of the matrix from the host buffer's data.
  void update_host_mirror() const {
#ifdef HPGMP_WITH_CUDA
      if(cudaSuccess != cudaMemcpy(values_, d_values_, m_*n_*sizeof(scalar_type),
                                   cudaMemcpyDeviceToHost)) {
          throw HostDeviceCopyFailedError("Failed to update host mirror!");
      }
#elif defined HPGMP_WITH_HIP
      if(hipSuccess != hipMemcpy(values_, d_values_, m_*n_*sizeof(scalar_type),
                                 hipMemcpyDeviceToHost)) {
          throw HostDeviceCopyFailedError("Failed to update host mirror!");
      }
#endif
  }
  
  // aux for profile
  double time1, time2;

private:
  local_int_t m_;        //!< number of rows
  local_int_t n_;        //!< number of columns

  SC * values_;          //!< array of values
//#if defined(HPGMP_WITH_CUDA) | defined(HPGMP_WITH_HIP)
  SC * d_values_;        //!< array of values
//#endif
  DeviceCtx *dctx_;
  /*!
   This is for storing optimized data structures created in OptimizeProblem and
   used inside optimized ComputeSPMV().
   */
  void * optimizationData_ = nullptr;
};


/*
  Copy input matrix to output matrix.

  @param[in] A Input vector
  @param[in] B Output vector
 */
//template<class SerialDenseMatrix_type>
//inline void CopyMatrix(const SerialDenseMatrix_type & A, SerialDenseMatrix_type & B) {
//
//  typedef typename SerialDenseMatrix_type::scalar_type scalar_type;
//
//  local_int_t m = A.m;
//  local_int_t n = A.n;
//  assert(B.m >= m);
//  assert(A.n >= n);
//  scalar_type * val_in  = A.values;
//  scalar_type * val_out = B.values;
//  for (int i=0; i<m*n; ++i)
//    val_out[i] = val_in[i];
//  return;
//}

#endif // SERIAL_DENSE_MATRIX_HPP
