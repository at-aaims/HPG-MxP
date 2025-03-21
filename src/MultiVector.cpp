
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
#include "MultiVector.hpp"

#include <cassert>
#include <cstdlib>

#include "DataTypes.hpp"

template<class scalar>
MultiVector<scalar>::MultiVector() { }

template<class scalar>
MultiVector<scalar>::MultiVector(const local_int_t localLength, const local_int_t n,
                                 comm_type comm, DeviceCtx *const dctx)
{
    initialize(localLength, n, comm, dctx);
}

template<class scalar>
MultiVector<scalar>::MultiVector(const local_int_t localLength, const local_int_t n,
                                 comm_type comm, DeviceCtx *const dctx, scalar *const vals,
                                 scalar *const d_vals)
{
    initialize_view(localLength, n, comm, dctx, vals, d_vals);
}

template<class scalar>
void MultiVector<scalar>::initialize(const local_int_t localLength, const local_int_t n,
                                     comm_type comm, DeviceCtx *const dctx)
{
  if(localLength_ != 0) {
      throw std::runtime_error("Invalid reinitialization of multivector!");
  }
  localLength_ = localLength;
  n_ = n;
  comm_ = comm;
  dctx_ = dctx;
  //#if defined(HPGMP_WITH_CUDA)
  //if (cudaSuccess != cudaMalloc ((void**)&d_values_, (localLength*n)*sizeof(scalar))) {
  //  throw DeviceMemoryError("Cuda malloc failed for multivector!");
  //}
  //#elif defined(HPGMP_WITH_HIP)
  //if (hipSuccess != hipMalloc ((void**)&d_values_, (localLength*n)*sizeof(scalar))) {
  //  throw DeviceMemoryError("HIP malloc failed for multivector!");
  //}
  //#endif
#if defined(HPGMP_WITH_CUDA) || defined(HPGMP_WITH_HIP)
  d_values_ = (scalar*)dctx_->device_alloc(localLength_*n_*sizeof(scalar));
  values_ = (scalar*)dctx_->pinned_host_alloc(localLength_*n_*sizeof(scalar));
#else
  values_ = new scalar[localLength * n];
#endif
  is_view_ = false;
}

template<class scalar>
void MultiVector<scalar>::initialize_view(const local_int_t localLength, const local_int_t n,
                                          comm_type comm, DeviceCtx *const dctx,
                                          scalar *const values, scalar *const d_values)
{
  if(localLength_ != 0) {
      throw std::runtime_error("Invalid reinitialization of multivector!");
  }
  localLength_ = localLength;
  n_ = n;
  values_ = values;
  d_values_ = d_values;
  comm_ = comm;
  dctx_ = dctx;
  is_view_ = true;
}

template <typename scalar>
void MultiVector<scalar>::fill_zero()
{
    const int zero (0);
#ifdef HPGMP_WITH_CUDA
    if (cudaSuccess != cudaMemset(d_values_, zero, localLength_*n_*sizeof(scalar))) {
        throw DeviceMemoryError("Cuda memset failed for multivector!");
    }
#elif defined(HPGMP_WITH_HIP)
    if (hipSuccess != hipMemset(d_values_, zero, localLength_*n_*sizeof(scalar))) {
        throw DeviceMemoryError("HIP memset failed for multivector!");
    }
#else
    for (int i=0; i<localLength_*n_; ++i) {
        values_[i] = zero;
    }
#endif
}

template<typename scalar>
Vector<scalar> MultiVector<scalar>::get_vector(const local_int_t jvec)
{
    return Vector<scalar>(localLength_, comm_, dctx_, values_ + localLength_*jvec,
                          d_values_ + localLength_*jvec);
}

template<typename scalar>
MultiVector<scalar> MultiVector<scalar>::get_multi_vector(const local_int_t j1, const local_int_t j2)
{
    return MultiVector<scalar>(localLength_, j2-j1+1, comm_, dctx_, values_ + localLength_*j1,
                               d_values_ + localLength_*j1);
}

template <typename scalar>
void MultiVector<scalar>::update_host_mirror() const
{
#ifdef HPGMP_WITH_CUDA
    if(cudaSuccess != cudaMemcpy(values_, d_values_, localLength_*n_*sizeof(scalar),
                                 cudaMemcpyDeviceToHost)) {
        throw HostDeviceCopyFailedError("Failed to update host mirror!");
    }
#elif HPGMP_WITH_HIP
    if(hipSuccess != hipMemcpy(values_, d_values_, localLength_*n_*sizeof(scalar),
                               hipMemcpyDeviceToHost)) {
        throw HostDeviceCopyFailedError("Failed to update host mirror!");
    }
#endif
}

template <typename scalar>
void MultiVector<scalar>::update_device_data() const
{
#ifdef HPGMP_WITH_CUDA
    if(cudaSuccess != cudaMemcpy(d_values_, values_, localLength_*n_*sizeof(scalar),
                                 cudaMemcpyHostToDevice)) {
        throw HostDeviceCopyFailedError("Failed to update device data!");
    }
#elif HPGMP_WITH_HIP
    if(hipSuccess != hipMemcpy(d_values_, values_, localLength_*n_*sizeof(scalar),
                               hipMemcpyHostToDevice)) {
        throw HostDeviceCopyFailedError("Failed to update device data!");
    }
#endif
}

/*!
  Deallocates the members of the data structure of the known system matrix provided they are not 0.

  @param[in] A the known system matrix
 */
template<class scalar>
MultiVector<scalar>::~MultiVector()
{
    if(!is_view_) {
#if defined(HPGMP_WITH_CUDA) || defined(HPGMP_WITH_HIP)
        dctx_->device_free(d_values_);
        dctx_->pinned_host_free(values_);
#else
        delete [] values_;
#endif
    }
    d_values_ = values_ = nullptr;
    localLength_ = 0;
    n_ = 0;
}

template class MultiVector<double>;
template class MultiVector<float>;

