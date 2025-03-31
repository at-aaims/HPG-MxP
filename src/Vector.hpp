
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

 HPGMP routine
 */

#ifndef HPGMP_VECTOR_HPP
#define HPGMP_VECTOR_HPP

#include <fstream>
#include <cassert>
#include <cstdlib>
#include <stdexcept>

#ifdef HPGMP_WITH_BLAS
 #include "cblas.h"
#endif

#ifndef HPGMP_NO_MPI
#include <vector>
#include <mpi.h>
#endif

#include "device_ctx.hpp"
#include "hpgmp.hpp"
#include "matrix_base.hpp"

template<class SC = double>
class Vector {
public:
  typedef SC scalar_type;

  /** Creates uninitialized vector.
   *
   * @sa initialize
   */
  Vector();

  /** Creates a vector.
   *
   * @param local_length  Length of local part of the vector.
   * @param comm  MPI communicator associated with the distributed vector.
   * @param device_ctx  Context of the device (accelerator/CPU) on which this vector resides.
   */
  Vector(local_int_t localLength, comm_type comm, DeviceCtx *dev_ctx);

  /** Creates a vector view.
   *
   * @param local_length  Length of local part of the vector.
   * @param comm  MPI communicator associated with the distributed vector.
   * @param device_ctx  Context of the device (accelerator/CPU) on which this vector resides.
   */
  Vector(local_int_t localLength, comm_type comm, DeviceCtx *dev_ctx, SC *values, SC *d_values);

  ~Vector();
 
  /// Initialize the vector. 
  void initialize(local_int_t localLength, comm_type comm, DeviceCtx *dev_ctx);

  /// Initialize a non-managing view into a different entity.
  void initialize_view(local_int_t localLength, comm_type comm, DeviceCtx *dev_ctx,
                       SC *values, SC *d_values);

  SC* values() { return values_; }
  SC* d_values() { return d_values_; }
  const SC* values() const { return values_; }
  const SC* d_values() const { return d_values_; }
  local_int_t local_length() const { return localLength_; }

  DeviceCtx *get_device_context() const { return dctx_; }
  comm_type get_comm() const { return comm_; }

  dev_blas_ctx get_blas_handle() const { return dctx_->get_blas_handle(); }

  /// Updates the host copy of the vector from device data.
  void update_host_mirror() const;

  /// Updates the device data of the vector from the host buffer's data.
  void update_device_data() const;

  /** Blocking halo update of distributed vector
   * according to the discretization represented by a matrix.
   */
  void update_halos(const DistMatrixBase *mat) const;
 
  /// Asynchronous packing of send buffer on the halo stream 
  void update_halos_pack_send_buffer(const DistMatrixBase *mat) const;
  
  /// Issue asynchronous send and receive commands, synchronizing the halo stream
  void update_halos_send_receive(const DistMatrixBase *mat) const;

  /// Wait for the communications and send data to GPU if necessary.
  void update_halos_finalize(const DistMatrixBase *mat) const;

  // Some operations

  /// Fill vector with zero values
  void fill_zero();

  /// Fills vector with random values between 1 and 2.
  void fill_random();

  /// Scales the vector by multiplying with a scalar.
  void scale(scalar_type value);
  
  mutable double time1{}, time2{}, time3{}, time4{};

private:
  local_int_t localLength_ = 0;  //!< length of local portion of the vector

  /// communicator
  comm_type comm_;

  /// Device context for this vector
  DeviceCtx *dctx_;

  SC * values_ = nullptr;     //!< array of values

  SC * d_values_ = nullptr;   //!< array of values
  bool is_view_ = false;

#ifndef HPGMP_NO_MPI
  mutable std::vector<MPI_Request> send_reqs_;
  mutable std::vector<MPI_Request> recv_reqs_;
  mutable bool halos_buffer_packed_ = false;
#endif

  /*!
   This is for storing optimized data structures created in OptimizeProblem and
   used inside optimized ComputeSPMV().
   */
  void * optimizationData = nullptr;

  mutable double t0_;
};

/*!
 * Copy input vector to output vector.
 *
 * @param[in] v Input vector
 * @param[in] w Output vector
 */
template<class scalar_src, class scalar_dst>
void CopyVector(const Vector<scalar_src> & v, Vector<scalar_dst> & w);

#endif // VECTOR_HPP
