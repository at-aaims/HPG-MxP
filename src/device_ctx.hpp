#ifndef HPGMP_DEVICE_CTX_HPP
#define HPGMP_DEVICE_CTX_HPP

#include "DataTypes.hpp"

#ifdef HPGMP_WITH_HIP
using dev_blas_ctx = rocblas_handle;
#elif defined HPGMP_WITH_CUDA
using dev_blas_ctx = cublasHandle_t;
#else
using dev_blas_ctx = int;
#endif

#ifdef HPGMP_WITH_HIP
using dev_spblas_ctx = rocsparse_handle;
#elif defined HPGMP_WITH_CUDA
using dev_spblas_ctx = cusparseHandle_t;
#else
using dev_spblas_ctx = int;
#endif

#ifdef HPGMP_WITH_HIP
using stream_t = hipStream_t;
using event_t  = hipEvent_t;
#elif defined HPGMP_WITH_CUDA
using stream_t = cudaStream_t;
using event_t  = cudaEvent_t;
#else
struct stream_t
{ };
struct event_t
{ };
#endif

/**
 * A context class to manage the accelerator device context independent of type.
 *
 * Only one object of this type is allowed per process per thread.
 *
 * WARNING: The functionality in this class is not thread-safe.
 * It is assumed that an object is created, used and destroyed only by one, fixed thread.
 */
class DeviceCtx
{
public:
    DeviceCtx(int process_rank);
    ~DeviceCtx();

    dev_blas_ctx get_blas_handle() { return blas_handle_; }
    dev_spblas_ctx get_sparse_handle() { return sparse_handle_; }
    stream_t get_compute_stream() { return compute_stream_; }
    stream_t get_halo_stream() { return halo_stream_; }

    /// Allocate storage on device
    void* device_alloc(size_t bytes);

    /// Free storage pointed to by a device pointer
    void device_free(void* ptr);

    /// Allocate pinned memory on the host
    void* pinned_host_alloc(size_t bytes);

    /// Free pinned host memory
    void pinned_host_free(void* ptr);

    /// Sync copy to device
    void copy_host_to_device_sync(void* d_ptr, const void* h_ptr, size_t nbytes);

    /// Sync copy to host
    void copy_device_to_host_sync(void* h_ptr, const void* d_ptr, size_t nbytes);

    /// Sync copy from one device buffer to another.
    void copy_device_to_device_sync(void* dst_ptr, const void* src_ptr, size_t nbytes);

    /// Asynchronous copy to device
    void copy_host_to_device_async(void* d_ptr, const void* h_ptr, size_t nbytes, stream_t stream);

    /// Asynchronous copy to host
    void copy_device_to_host_async(void* h_ptr, const void* d_ptr, size_t nbytes, stream_t stream);

    /// Synchronize compute stream with host
    void synchronize_compute_stream();

    /// Synchronize halo stream with host
    void synchronize_halo_stream();

    /// Synchronize entire device
    void synchronize_device();

    /// Set memory to repeated constant integer value
    void device_memset(void* d_ptr, int value, size_t nbytes);

    /// Create default-type event
    event_t create_event();

    void destroy_event(event_t event);

    /// Create event which "happens" when all previous commands are complete.
    void record_event(event_t event, stream_t stream);

    /// Make the given stream wait for the given event.
    void stream_wait_on_event(stream_t stream, event_t event);

    /// Get a pre-allocted workspace on the device
    void* get_device_workspace() const { return workspace_; }

private:
    stream_t halo_stream_;
    stream_t compute_stream_;
    dev_blas_ctx blas_handle_;
    dev_spblas_ctx sparse_handle_;
    const int rank_;
    void* workspace_ = nullptr;
    /// Keep track of whether an object has already been created.
    static bool is_instantiated;
};

/// Padding multiple for GPU allocation of multi-vectors.
template<typename T>
struct padding_multiple
{
#ifdef HPGMP_WITH_GINKGO
    static constexpr int value = 128; // Ginkgo expects the same size for indices and values
#else
    static constexpr int value = static_cast<int>(128 / sizeof(T));
#endif
};


#endif
