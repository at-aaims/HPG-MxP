
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
// Aditya Kashi              (kashia@ornl.gov)
//
// ***************************************************
//@HEADER

#include "Vector.hpp"

#include <cassert>
#include <string>
#include <iostream>

#ifdef HPGMP_WITH_HIP
#include <rocrand/rocrand.hpp>
#elif defined HPGMP_WITH_CUDA
#include <curand.h>
#endif

#include "Utils_MPI.hpp"
#include "mytimer.hpp"
#include "exceptions.hpp"

template <typename scalar>
Vector<scalar>::Vector()
{ }
 
template <typename scalar>
Vector<scalar>::Vector(local_int_t localLength, comm_type comm, DeviceCtx *const dev_ctx)
{
    initialize(localLength, comm, dev_ctx);
}
 
template <typename scalar>
void Vector<scalar>::initialize(local_int_t localLength, comm_type comm, DeviceCtx *dev_ctx)
{
    is_view_ = false;
    if(localLength_ != 0) {
        throw std::runtime_error("Invalid reinitialize!");
    }
    localLength_ = localLength;
    comm_ = comm;
    dctx_ = dev_ctx;
#if defined(HPGMP_WITH_CUDA) || defined(HPGMP_WITH_HIP)
    d_values_ = (scalar*)dctx_->device_alloc(localLength*sizeof(scalar));
    values_ = (scalar*)dctx_->pinned_host_alloc(localLength*sizeof(scalar));
    send_gather_ = dctx_->create_event();
#else
    v.values = new scalar_type[localLength];
#endif
}
  
template <typename scalar>
Vector<scalar>::Vector(local_int_t localLength, comm_type comm, DeviceCtx *dev_ctx,
                       scalar *values, scalar *d_values)
{
    initialize_view(localLength, comm, dev_ctx, values, d_values);
}
  
template <typename scalar>
void Vector<scalar>::initialize_view(local_int_t localLength, comm_type comm, DeviceCtx *dev_ctx,
                                     scalar *v, scalar *d_v)
{
    is_view_ = true;
    if(localLength_ != 0) {
        throw std::runtime_error("Invalid reinitialize!");
    }
    localLength_ = localLength;
    comm_ = comm;
    dctx_ = dev_ctx;
    d_values_ = d_v;
    values_ = v;
#if defined(HPGMP_WITH_CUDA) || defined(HPGMP_WITH_HIP)
    send_gather_ = dctx_->create_event();
#endif
}

template <typename scalar>
void Vector<scalar>::fill_zero()
{
    const int zero (0);
#ifdef HPGMP_WITH_CUDA
    if (cudaSuccess != cudaMemset(d_values_, zero, localLength_*sizeof(scalar))) {
        printf( " CopyVector :: Failed to memcpy d_v\n" );
    }
#elif defined(HPGMP_WITH_HIP)
    if (hipSuccess != hipMemset(d_values_, zero, localLength_*sizeof(scalar))) {
        printf( " CopyVector :: Failed to memcpy d_v\n" );
    }
#else
    for (int i=0; i<localLength_; ++i) {
        values_[i] = zero;
    }
#endif
}

template<class scalar>
void Vector<scalar>::fill_random() {
#ifdef HPGMP_WITH_HIP
  rocrand_generator generator;
  rocrand_create_generator(&generator, ROCRAND_RNG_PSEUDO_DEFAULT);
  rocrand_set_stream(generator, dctx_->get_compute_stream());
  rocrand_initialize_generator(generator);
  //const rocblas_datatype alpha_type = rocblas_datatype_f64_r;
  //rocblas_datatype x_type;
  //auto handle = dctx_->get_blas_handle();
  rocrand_status status = ROCRAND_STATUS_SUCCESS;
  if constexpr(std::is_same<scalar,double>::value) {
      status = rocrand_generate_uniform_double(generator, d_values_, localLength_);
      //x_type = rocblas_datatype_f_64_r;
  } else if constexpr(std::is_same<scalar,float>::value) {
      status = rocrand_generate_uniform(generator, d_values_, localLength_);
      //x_type = rocblas_datatype_f_32_r;
  } else {
      throw std::runtime_error("fill_random: Unsupported scalar type");
  }
  if(status != ROCRAND_STATUS_SUCCESS) {
      throw std::runtime_error("random vector generation failed!");
  }
  dctx_->synchronize_compute_stream();
  // TODO: Add 1 to generated vector
  rocrand_destroy_generator(generator);
#elif defined HPGMP_WITH_CUDA
  curandGenerator_t generator;
  curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetStream(generator, dctx_->get_compute_stream());
  curandStatus status = CURAND_STATUS_SUCCESS;
  if constexpr(std::is_same<scalar,double>::value) {
      status = curandGenerateUniformDouble(generator, d_values_, localLength_);
  } else if constexpr(std::is_same<scalar,float>::value) {
      status = curandGenerateUniform(generator, d_values_, localLength_);
  } else {
      throw std::runtime_error("fill_random: Unsupported scalar type");
  }
  if(status != CURAND_STATUS_SUCCESS) {
      throw std::runtime_error("random vector generation failed!");
  }
  dctx_->synchronize_compute_stream();
  // TODO: Add 1 to generated vector
  curandDestroyGenerator(generator);
#else
  scalar * vv = v.values();
  for (int i=0; i<localLength_; ++i) {
      vv[i] = rand() / (scalar)(RAND_MAX) + 1.0;
  }
#endif
}

template <typename scalar_type>
void Vector<scalar_type>::scale(const scalar_type value)
{
  auto handle = dctx_->get_blas_handle();

#if defined(HPGMP_WITH_CUDA) || defined(HPGMP_WITH_HIP)
  auto d_vv = d_values_;
  if constexpr(std::is_same<scalar_type, double>::value) {
    #ifdef HPGMP_WITH_CUDA
    if (CUBLAS_STATUS_SUCCESS != cublasDscal (handle, localLength_, (const double*)&value, (double*)d_vv, 1)) {
      printf( " Failed cublasDscal\n" );
    }
    #elif defined(HPGMP_WITH_HIP)
    if (rocblas_status_success != rocblas_dscal (handle, localLength_, (const double*)&value, (double*)d_vv, 1)) {
      printf( " Failed rocblas_dscal\n" );
    }
    #endif
  } else if constexpr(std::is_same<scalar_type, float>::value) {
    #ifdef HPGMP_WITH_CUDA
    if (CUBLAS_STATUS_SUCCESS != cublasSscal (handle, localLength_, (const float*)&value, (float*)d_vv, 1)) {
      printf( " Failed cublasSscal\n" );
    }
    #elif defined(HPGMP_WITH_HIP)
    if (rocblas_status_success != rocblas_sscal (handle, localLength_, (const float*)&value, (float*)d_vv, 1)) {
      printf( " Failed rocblas_sscal\n" );
    }
    #endif
  }
#else
  // host CPU
  scalar_type * vv = v.values();
  #if defined(HPGMP_WITH_BLAS)
  if (std::is_same<scalar_type, double>::value) {
    cblas_dscal(localLength_, value, (double *)vv, 1);
  } else if (std::is_same<scalar_type, float>::value) {
    cblas_sscal(localLength_, value, (float *)vv, 1);
  }
  #else
  const scalar_type zero (0.0);
  if (value == zero) {
    for (int i=0; i<localLength_; ++i)
        vv[i] = zero;
  } else {
    for (int i=0; i<localLength_; ++i)
        vv[i] = value * scalar_type(vv[i]);
  }
  #endif
#endif
  return;
}

template <typename scalar>
void Vector<scalar>::update_host_mirror() const
{
#ifdef HPGMP_WITH_CUDA
    if(cudaSuccess != cudaMemcpy(values_, d_values_, localLength_*sizeof(scalar),
                                cudaMemcpyDeviceToHost)) {
        throw HostDeviceCopyFailedError("Failed to update host mirror!");
    }
#elif HPGMP_WITH_HIP
    if(hipSuccess != hipMemcpy(values_, d_values_, localLength_*sizeof(scalar),
                              hipMemcpyDeviceToHost)) {
        throw HostDeviceCopyFailedError("Failed to update host mirror!");
    }
#endif
}

template <typename scalar>
void Vector<scalar>::update_device_data() const
{
#ifdef HPGMP_WITH_CUDA
    if(cudaSuccess != cudaMemcpy(d_values_, values_, localLength_*sizeof(scalar),
                                cudaMemcpyHostToDevice)) {
        throw HostDeviceCopyFailedError("Failed to update device data!");
    }
#elif HPGMP_WITH_HIP
    if(hipSuccess != hipMemcpy(d_values_, values_, localLength_*sizeof(scalar),
                              hipMemcpyHostToDevice)) {
        throw HostDeviceCopyFailedError("Failed to update device data!");
    }
#endif
}

template <typename scalar>
Vector<scalar>::~Vector() {
    if(!is_view_) {
#if defined(HPGMP_WITH_CUDA) || defined(HPGMP_WITH_HIP)
        dctx_->device_free(d_values_);
        dctx_->pinned_host_free(values_);
        dctx_->destroy_event(send_gather_);
#else
        delete [] values_;
#endif
    }
    d_values_ = values_ = nullptr;
    localLength_ = 0;
}

namespace {

#if defined(HPGMP_WITH_CUDA) || defined(HPGMP_WITH_HIP)

template <typename scalar>
__global__ void haloGather(const int totalToBeSent, const scalar *const d_x,
                           const int *const d_elementsToSend, const local_int_t *const perm,
                           scalar *const d_sendBuffer)
{
  const int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i < totalToBeSent) {
    d_sendBuffer[i] = d_x[perm[d_elementsToSend[i]]];
  }
}

#endif

}

#ifdef HPGMP_ONLY_BLOCKING_COMMS

template <typename scalar>
void Vector<scalar>::update_halos_pack_send_buffer(const DistMatrixBase *const mat) const
{
#ifndef HPGMP_NO_MPI
    const MPI_Datatype MPI_SCALAR_TYPE = MpiTypeTraits<scalar>::getType ();
    const local_int_t localNumberOfRows = mat->get_local_num_rows();
    const local_int_t localNumberOfCols = mat->get_local_num_cols();
    const int num_neighbors = mat->get_num_neighbors();
    const local_int_t *const receiveLength = mat->get_receive_lengths();
    const local_int_t *const sendLength = mat->get_send_lengths();
    const int *const neighbors = mat->get_neighbors();
    const local_int_t totalToBeSent = mat->get_total_to_be_sent();

    assert(localNumberOfRows + totalToBeSent <= localLength_);

#ifndef HPGMP_USE_GPU_AWARE_MPI
    scalar_type * const xv = values_;
#endif
    scalar_type * const d_xv = d_values_;

    int size, rank; // Number of MPI processes, My process ID
    auto comm = mat->get_comm();
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    if (size == 1)
        return;

    const int MPI_MY_TAG = 99;

    //
    // Externals are at end of locals
    //
#ifdef HPGMP_USE_GPU_AWARE_MPI
    scalar_type * x_external = (scalar_type *) d_xv + localNumberOfRows;
#else
    scalar_type * x_external = (scalar_type *) xv + localNumberOfRows;
#endif

    auto halo_stream = dctx_->get_halo_stream();
    
    recv_reqs_.resize(num_neighbors);

    // Post receives first
    // Thread this loop
    double t0 = 0.0;
    TICK();
    for (int i = 0; i < num_neighbors; i++) {
        const local_int_t n_recv = receiveLength[i];
        MPI_Irecv(x_external, n_recv, MPI_SCALAR_TYPE, neighbors[i], MPI_MY_TAG, comm, &recv_reqs_[i]);
        x_external += n_recv;
    }
    double time2 = 0.0;
    TOCK(time2);


    // Fill up send buffer
    TICK();
    auto d_sendBuffer = static_cast<scalar_type*>(mat->get_device_send_buffer());
    const auto perm = mat->get_reordering_permutation();
    const auto d_elemstosend = mat->get_elements_to_send();

#if defined HPGMP_WITH_HIP || defined HPGMP_WITH_CUDA
    const int num_threads = (totalToBeSent < 256 ? totalToBeSent : 256);
    const int num_blocks = (totalToBeSent+num_threads-1)/num_threads;
    haloGather<<<num_blocks, num_threads, 0, halo_stream>>>(
        totalToBeSent, d_xv, d_elemstosend, perm, d_sendBuffer);
#else
    for(local_int_t i = 0; i < totalToBeSent; i++) {
        d_sendBuffer[i] = d_xv[perm[d_elemstosend[i]]];
    }
#endif

#ifdef HPGMP_USE_GPU_AWARE_MPI
    auto sendBuffer = d_sendBuffer;
#else
    auto sendBuffer = static_cast<scalar*>(mat->get_host_send_buffer());
    dctx_->copy_device_to_host_async(sendBuffer, d_sendBuffer, totalToBeSent*sizeof(scalar_type),
                                     halo_stream);

#endif

    // wait for buffer to be ready before sending
    dctx_->synchronize_halo_stream();

    double time1 = 0.0;
    TOCK(time1);

    // Send to each neighbor
    TICK();
    for (int i = 0; i < num_neighbors; i++) {
        local_int_t n_send = sendLength[i];
        MPI_Send(sendBuffer, n_send, MPI_SCALAR_TYPE, neighbors[i], MPI_MY_TAG, comm);
        sendBuffer += n_send;
    }

    // Complete the recvs issued above
    if(MPI_SUCCESS != MPI_Waitall(num_neighbors, recv_reqs_.data(), MPI_STATUSES_IGNORE)) {
        printf(" Vector: update_halo: receives failed!"); fflush(stdout);
        MPI_Abort(comm, -2025);
    }
    TOCK(time2);

#if !defined(HPGMP_USE_GPU_AWARE_MPI)
    // copy received data to GPU
    TICK();
    dctx_->copy_host_to_device_async(d_xv + localNumberOfRows, xv + localNumberOfRows,
                                     (localNumberOfCols-localNumberOfRows)*sizeof(scalar_type),
                                     halo_stream);
    TOCK(time1);
#endif
    recv_reqs_.clear();
#endif // HPGMP_NO_MPI
}

template <typename scalar>
void Vector<scalar>::update_halos_send_receive(const DistMatrixBase *const mat) const { }

template <typename scalar>
void Vector<scalar>::update_halos_finalize(const DistMatrixBase *const mat) const { }

#else // HPGMP_ONLY_BLOCKING_COMMS

template <typename scalar>
void Vector<scalar>::update_halos_pack_send_buffer(const DistMatrixBase *const mat) const
{
#ifndef HPGMP_NO_MPI
    const local_int_t totalToBeSent = mat->get_total_to_be_sent();
    scalar_type * const d_xv = d_values_;
    auto d_sendBuffer = static_cast<scalar_type*>(mat->get_device_send_buffer());
    const auto perm = mat->get_reordering_permutation();
    const auto d_elemstosend = mat->get_elements_to_send();
    auto halo_stream = dctx_->get_halo_stream();
    auto interior_stream = dctx_->get_compute_stream();

    // Fill up send buffer
    // timing
    TICK_STREAM_SYNC(halo_stream, t0_);
#if defined HPGMP_WITH_HIP || defined HPGMP_WITH_CUDA
    const int num_threads = (totalToBeSent < 256 ? 512 : 256);
    const int num_blocks = (totalToBeSent - 1)/num_threads + 1;
    haloGather<<<num_blocks, num_threads, 0, halo_stream>>>(
        totalToBeSent, d_xv, d_elemstosend, perm, d_sendBuffer);
    // define an event that waits for the halo gather kernel to complete
    dctx_->record_event(send_gather_,  halo_stream);
#else
    for(local_int_t i = 0; i < totalToBeSent; i++) {
        d_sendBuffer[i] = d_xv[perm[d_elemstosend[i]]];
    }
#endif

#if !defined(HPGMP_USE_GPU_AWARE_MPI)
    auto sendBuffer = static_cast<scalar*>(mat->get_host_send_buffer());
    dctx_->copy_device_to_host_async(sendBuffer, d_sendBuffer, totalToBeSent*sizeof(scalar_type),
                                     halo_stream);
#endif
    halos_buffer_packed_ = true;

    // make it so that any future work submitted to the compute stream
    //   waits to execute till the send buffer is gathered
    dctx_->stream_wait_on_event(interior_stream, send_gather_);
#endif // HPGMP_NO_MPI
}

template <typename scalar>
void Vector<scalar>::update_halos_send_receive(const DistMatrixBase *const mat) const
{
#ifndef HPGMP_NO_MPI
    const MPI_Datatype MPI_SCALAR_TYPE = MpiTypeTraits<scalar>::getType ();
    const local_int_t localNumberOfRows = mat->get_local_num_rows();
    const local_int_t totalToBeSent = mat->get_total_to_be_sent();
    const int num_neighbors = mat->get_num_neighbors();
    const local_int_t *const receiveLength = mat->get_receive_lengths();
    const local_int_t *const sendLength = mat->get_send_lengths();
    const int *const neighbors = mat->get_neighbors();
    auto send_buffer =    
#ifdef HPGMP_USE_GPU_AWARE_MPI
        static_cast<scalar_type*>(mat->get_device_send_buffer());
#else
        static_cast<scalar_type*>(mat->get_host_send_buffer());
#endif
    auto halo_stream = dctx_->get_halo_stream();
    auto comm = mat->get_comm();

    assert(localNumberOfRows + totalToBeSent <= localLength_);

    int size, rank; // Number of MPI processes, My process ID
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    if (size == 1)
        return;
    
    const int MPI_MY_TAG = 99;

    // Externals are at end of locals

#ifdef HPGMP_USE_GPU_AWARE_MPI
    scalar_type * x_external = (scalar_type *) d_values_ + localNumberOfRows;
#else
    scalar_type * x_external = (scalar_type *) values_ + localNumberOfRows;
#endif
    
    recv_reqs_.resize(num_neighbors);

    // Post receives first
    for (int i = 0; i < num_neighbors; i++) {
        const local_int_t n_recv = receiveLength[i];
        MPI_Irecv(x_external, n_recv, MPI_SCALAR_TYPE, neighbors[i], MPI_MY_TAG, comm, &recv_reqs_[i]);
        x_external += n_recv;
    }

    // Ensure buffers are packed
    if(!halos_buffer_packed_) {
        throw std::runtime_error("Attempting to send without packing the send buffer!");
    }

    send_reqs_.resize(num_neighbors);

    // Wait for send-buffer packing and transfer, and record the time.
    TOCK_STREAM_SYNC(halo_stream, t0_, time1_);

    for (int i = 0; i < num_neighbors; i++) {
        const local_int_t n_send = sendLength[i];
        MPI_Isend(send_buffer, n_send, MPI_SCALAR_TYPE, neighbors[i], MPI_MY_TAG, comm, &send_reqs_[i]);
        send_buffer += n_send;
    }
#endif
}

namespace internal {

void check_waitall_error(const int ierr, const std::string& type)
{
    char error_msg[1024];
    int len{};
    int jerr = MPI_Error_string(ierr, error_msg, &len);
    if(jerr != MPI_SUCCESS) {
        std::cout << " Vector: update_halo: All " << type << "s failed! error = " << ierr
                  << std::endl;
    } else {
        std::cout << " Vector: update_halo: A " << type << " failed! error: " << error_msg
                  << std::endl;
    }
}

void check_waitall_statuses(const std::string&& type, const int ierr,
                            const std::vector<MPI_Status>& statuses)
{
    if(ierr != MPI_SUCCESS) {
        if(ierr == MPI_ERR_IN_STATUS) {
            for(unsigned i = 0; i < statuses.size(); i++) {
                check_waitall_error(statuses[i].MPI_ERROR, type);
                std::cout << " Vector: MPI tag " << statuses[i].MPI_TAG << ", source = "
                          << statuses[i].MPI_SOURCE << std::endl;
            }
        } else {
            check_waitall_error(ierr, type);
        }
        std::cout << std::flush;
        MPI_Abort(MPI_COMM_WORLD, -2025);
    }
}

}

template <typename scalar>
void Vector<scalar>::update_halos_finalize(const DistMatrixBase *const mat) const
{
#ifndef HPGMP_NO_MPI
    // Complete the sends and receives.
    const local_int_t localNumberOfRows = mat->get_local_num_rows();
    const local_int_t localNumberOfCols = mat->get_local_num_cols();
    const int num_neighbors = mat->get_num_neighbors();
    auto halo_stream = dctx_->get_halo_stream();
    std::vector<MPI_Status> send_statuses(num_neighbors), recv_statuses(num_neighbors);

    TICK_STREAM_SYNC(halo_stream, t0_);

    int ierr = MPI_Waitall(num_neighbors, recv_reqs_.data(), recv_statuses.data());
    internal::check_waitall_statuses("recv", ierr, recv_statuses);
    ierr = MPI_Waitall(num_neighbors, send_reqs_.data(), send_statuses.data());
    internal::check_waitall_statuses("send", ierr, send_statuses);
    
    // sync halo stream and record time taken for comms in time2.
    TOCK_STREAM_SYNC(halo_stream, t0_, time2_);

    TICK_STREAM_SYNC(halo_stream, t0_);
#if !defined(HPGMP_USE_GPU_AWARE_MPI)
    scalar_type * const xv = values_;
    scalar_type * const d_xv = d_values_;
    // copy received data to GPU - NOTE that the entire vector is not copied
    dctx_->copy_host_to_device_async(d_xv + localNumberOfRows, xv + localNumberOfRows,
                                     (localNumberOfCols-localNumberOfRows)*sizeof(scalar_type),
                                     halo_stream);
#endif
    // Sync halo stream and add to time taken for HD copies to time1.
    // Note that synchronization is necessary here to ensure received halos are available
    // for kernels on other streams.
    TOCK_STREAM_SYNC(halo_stream, t0_, time1_);

    halos_buffer_packed_ = false;
    recv_reqs_.clear();
    send_reqs_.clear();
#endif
}

#endif


template <unsigned int BLOCKSIZE, typename scalar>
__launch_bounds__(BLOCKSIZE)
__global__ void kernel_permute(const local_int_t size,
                               const local_int_t* __restrict__ perm,
                               const scalar* __restrict__ in,
                               scalar* __restrict__ out)
{
    const local_int_t gid = blockIdx.x * BLOCKSIZE + threadIdx.x;
    if(gid >= size) {
        return;
    }
    out[perm[gid]] = in[gid];
}

template <typename scalar>
void Vector<scalar>::permute(const local_int_t *const perm)
{
    const auto size = localLength_;
    auto buffer = reinterpret_cast<scalar*>(dctx_->device_alloc(size*sizeof(scalar)));

    kernel_permute<1024><<<(size - 1) / 1024 + 1, 1024>>>(size, perm, d_values_, buffer);

    dctx_->device_free(d_values_);
    d_values_ = buffer;
}

// Explicit instantiations
// TODO: add half
template class Vector<double>;
template class Vector<float>;

//template void Vector<double>::scale(double);
//template void Vector<double>::scale(float);
//template void Vector<float>::scale(double);
//template void Vector<float>::scale(float);


#if defined(HPGMP_WITH_CUDA) || defined(HPGMP_WITH_HIP)
template <typename scalar_src, typename scalar_dst>
__global__
void mxp_copy(const scalar_src *const __restrict__ src, scalar_dst *const __restrict__ dst, const int len)
{
    const int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if(tid < len) {
        dst[tid] = static_cast<scalar_dst>(src[tid]);
    }
}
#endif

/*!
  Copy input vector to output vector.

  @param[in] v Input vector
  @param[in] w Output vector
 */
template<class scalar_src, class scalar_dst>
void CopyVector(const Vector<scalar_src>& v, Vector<scalar_dst>& w)
{
  auto dctx = v.get_device_context();
  const local_int_t localLength = v.local_length();
  assert(w.local_length() >= localLength);
#if (!defined(HPGMP_WITH_CUDA) & !defined(HPGMP_WITH_HIP)) | defined(HPGMP_DEBUG)
  const scalar_src * vv = v.values();
  scalar_dst * wv = w.values();
  #if defined(HPGMP_WITH_BLAS)
  if (std::is_same<scalar_src, scalar_dst>::value) {
    if (std::is_same<scalar_src, double>::value) {
      cblas_dcopy(localLength, (const double *)vv, 1, (double *)wv, 1);
    } else if (std::is_same<scalar_src, float>::value) {
      cblas_scopy(localLength, (const float *)vv, 1, (float *)wv, 1);
    }
  } else
  #endif
  {
    for (int i=0; i<localLength; ++i)
        wv[i] = vv[i];
  }
#endif

#if defined(HPGMP_WITH_CUDA) | defined(HPGMP_WITH_HIP)
  if (std::is_same<scalar_src, scalar_dst>::value) {
    #ifdef HPGMP_DEBUG
    HPGMP_fout << " CopyVector ( Unit-precision )" << std::endl;
    #endif
    #if defined(HPGMP_WITH_CUDA)
    if (cudaSuccess != cudaMemcpy(w.d_values(), v.d_values(), localLength*sizeof(scalar_src),
                                  cudaMemcpyDeviceToDevice)) {
      printf( " CopyVector :: Failed to memcpy d_x\n" );
    }
    #elif defined(HPGMP_WITH_HIP)
    if (hipSuccess != hipMemcpy(w.d_values(), v.d_values(), localLength*sizeof(scalar_src),
                                hipMemcpyDeviceToDevice)) {
      printf( " CopyVector :: Failed to memcpy d_x\n" );
    }
    #endif
  } else {
    #ifdef HPGMP_DEBUG
      HPGMP_fout << " CopyVector ( mixed-precision )" << std::endl;
    #endif
      constexpr int block_size = 1024;
      const int nblocks = (localLength - 1) / block_size + 1;
      
      mxp_copy<<<nblocks, block_size, 0, dctx->get_compute_stream()>>>(
              v.d_values(), w.d_values(), localLength);
      dctx->synchronize_compute_stream();
  #if 0
    #ifdef HPGMP_DEBUG
    HPGMP_vout << " CopyVector :: Mixed-precision not supported" << std::endl;
    #endif
    // Copy input vector to Host
    #if defined(HPGMP_WITH_CUDA)
    if (cudaSuccess != cudaMemcpy(vv, v.d_values, localLength*sizeof(scalar_src), cudaMemcpyDeviceToHost)) {
      printf( " CopyVector :: Failed to memcpy d_v\n" );
    }
    #elif defined(HPGMP_WITH_HIP)
    if (hipSuccess != hipMemcpy(vv, v.d_values, localLength*sizeof(scalar_src), hipMemcpyDeviceToHost)) {
      printf( " CopyVector :: Failed to memcpy d_v\n" );
    }
    #endif

    // Copy on Host
    for (int i=0; i<localLength; ++i) wv[i] = vv[i];

    // Copy output vector to Device
    #if defined(HPGMP_WITH_CUDA)
    if (cudaSuccess != cudaMemcpy(w.d_values, wv, localLength*sizeof(scalar_dst), cudaMemcpyHostToDevice)) {
      printf( " CopyVector :: Failed to memcpy d_w\n" );
    }
    #elif defined(HPGMP_WITH_HIP)
    if (hipSuccess != hipMemcpy(w.d_values, wv, localLength*sizeof(scalar_dst), hipMemcpyHostToDevice)) {
      printf( " CopyVector :: Failed to memcpy d_w\n" );
    }
    #endif
  #endif
  }
#endif
}

template void CopyVector(const Vector<double>& v, Vector<float>& w);
template void CopyVector(const Vector<double>& v, Vector<double>& w);
template void CopyVector(const Vector<float>& v, Vector<float>& w);
template void CopyVector(const Vector<float>& v, Vector<double>& w);

