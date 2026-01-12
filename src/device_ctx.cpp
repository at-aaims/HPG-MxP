#include "device_ctx.hpp"

#include <iostream>
#include <cstdlib>
#include <new>

#include "hpgmp.hpp"
#include "exceptions.hpp"

#ifdef HPGMP_WITH_HIP
#define HPGMP_THROW_ON_ERROR(_expr, _msg) \
    if (hipSuccess != (_expr)) {          \
        throw std::runtime_error(_msg);   \
    }                                     \
    static_assert(true, "dummy assert for semicolon")
#define HPGMP_BLAS_THROW_ON_ERROR(_expr, _msg) \
    if (rocblas_status_success != (_expr)) {   \
        throw std::runtime_error(_msg);        \
    }                                          \
    static_assert(true, "dummy assert for semicolon")
#define HPGMP_SPARSE_THROW_ON_ERROR(_expr, _msg) \
    if (rocsparse_status_success != (_expr)) {   \
        throw std::runtime_error(_msg);          \
    }                                            \
    static_assert(true, "dummy assert for semicolon")

#define HPGMP_PRINT_ON_ERROR(_expr, _msg) \
    if (hipSuccess != (_expr)) {          \
        std::cout << (_msg) << std::endl; \
    }                                     \
    static_assert(true, "dummy assert for semicolon")
#define HPGMP_BLAS_PRINT_ON_ERROR(_expr, _msg) \
    if (rocblas_status_success != (_expr)) {   \
        std::cout << (_msg) << std::endl;      \
    }                                          \
    static_assert(true, "dummy assert for semicolon")
#define HPGMP_SPARSE_PRINT_ON_ERROR(_expr, _msg) \
    if (rocsparse_status_success != (_expr)) {   \
        std::cout << (_msg) << std::endl;        \
    }                                            \
    static_assert(true, "dummy assert for semicolon")
#else
#define HPGMP_THROW_ON_ERROR(_expr, _msg) static_assert(true, "dummy assert for semicolon")
#endif

bool DeviceCtx::is_instantiated = false;

DeviceCtx::DeviceCtx(const int process_rank)
    : rank_{process_rank}
{
    if (is_instantiated) {
        throw std::runtime_error("Should not instantiate more than 1 device context!");
    }
#if defined(HPGMP_WITH_CUDA)
    if (cudaSuccess != cudaStreamCreate(&halo_stream_)) {
        throw HandleNotCreatedError("halo stream");
    }
    if (cudaSuccess != cudaStreamCreate(&compute_stream_)) {
        throw HandleNotCreatedError("compute stream");
    }
    if (CUBLAS_STATUS_SUCCESS != cublasCreate(&blas_handle_)) {
        throw HandleNotCreatedError("cublas");
    }
    if (CUSPARSE_STATUS_SUCCESS != cusparseCreate(&sparse_handle_)) {
        throw HandleNotCreatedError("cusparse");
    }
#elif defined(HPGMP_WITH_HIP)
    if (hipSuccess != hipStreamCreate(&halo_stream_)) {
        throw HandleNotCreatedError("halo stream");
    }
    if (hipSuccess != hipStreamCreate(&compute_stream_)) {
        throw HandleNotCreatedError("compute stream");
    }
    if (rocblas_status_success != rocblas_create_handle(&blas_handle_)) {
        throw HandleNotCreatedError("rocblas");
    }
    if (rocsparse_status_success != rocsparse_create_handle(&sparse_handle_)) {
        throw HandleNotCreatedError("rocsparse");
    }
#endif
    if (rank_ == 0) {
        std::cout << "Created device context." << std::endl;
    }
    workspace_      = device_alloc(1024 * sizeof(local_int_t));
    is_instantiated = true;
}

DeviceCtx::~DeviceCtx()
{
#if defined(HPGMP_WITH_CUDA)
    cublasDestroy(blas_handle_);
    cusparseDestroy(sparse_handle_);
    cudaStreamDestroy(halo_stream_);
    cudaStreamDestroy(compute_stream_);
#elif defined(HPGMP_WITH_HIP)
    HPGMP_BLAS_PRINT_ON_ERROR(rocblas_destroy_handle(blas_handle_), "rocblas handle destroy");
    HPGMP_SPARSE_PRINT_ON_ERROR(rocsparse_destroy_handle(sparse_handle_),
                                "rocsparse handle destroy");
    HPGMP_PRINT_ON_ERROR(hipStreamDestroy(halo_stream_), "destroy halo stream");
    HPGMP_PRINT_ON_ERROR(hipStreamDestroy(compute_stream_), "destroy compute stream");
#endif
    device_free(workspace_);
    if (rank_ == 0) {
        std::cout << "Destroyed device context." << std::endl;
    }
}

void* DeviceCtx::device_alloc(const size_t bytes)
{
    void* ptr;
#if defined(HPGMP_WITH_CUDA)
    if (cudaSuccess != cudaMalloc(&ptr, bytes)) {
        throw std::bad_alloc();
    }
#elif defined(HPGMP_WITH_HIP)
    if (hipSuccess != hipMalloc(&ptr, bytes)) {
        throw std::bad_alloc();
    }
#else
    ptr = std::malloc(bytes);
#endif
    return ptr;
}

void DeviceCtx::device_free(void* ptr)
{
#if defined(HPGMP_WITH_CUDA)
    if (cudaSuccess != cudaFree(ptr)) {
        throw DeviceMemoryError("Could not cuda free!");
    }
#elif defined(HPGMP_WITH_HIP)
    if (hipSuccess != hipFree(ptr)) {
        throw DeviceMemoryError("Could not hip free!");
    }
#else
    std::free(ptr);
#endif
}

void* DeviceCtx::pinned_host_alloc(const size_t bytes)
{
    void* ptr;
#if defined(HPGMP_WITH_CUDA)
    if (cudaSuccess != cudaMallocHost(&ptr, bytes)) {
        throw std::bad_alloc();
    }
#elif defined(HPGMP_WITH_HIP)
    if (hipSuccess != hipHostMalloc(&ptr, bytes)) {
        throw std::bad_alloc();
    }
#else
    ptr = std::malloc(bytes);
#endif
    return ptr;
}

void DeviceCtx::pinned_host_free(void* ptr)
{
#if defined(HPGMP_WITH_CUDA)
    if (cudaSuccess != cudaFreeHost(ptr)) {
        throw DeviceMemoryError("Could not free Cuda pinned host memory!");
    }
#elif defined(HPGMP_WITH_HIP)
    if (hipSuccess != hipHostFree(ptr)) {
        throw DeviceMemoryError("Could not free HIP pinned host memory!");
    }
#else
    std::free(ptr);
#endif
}

void DeviceCtx::copy_host_to_device_sync(void* d_ptr, const void* h_ptr, size_t nbytes)
{
#if defined(HPGMP_WITH_CUDA)
    if (cudaSuccess != cudaMemcpy(d_ptr, h_ptr, nbytes, cudaMemcpyHostToDevice)) {
        throw HostDeviceCopyFailedError("H2D");
    }
#elif defined(HPGMP_WITH_HIP)
    if (hipSuccess != hipMemcpy(d_ptr, h_ptr, nbytes, hipMemcpyHostToDevice)) {
        throw HostDeviceCopyFailedError("H2D");
    }
#endif
}

void DeviceCtx::copy_device_to_host_sync(void* h_ptr, const void* d_ptr, size_t nbytes)
{
#if defined(HPGMP_WITH_CUDA)
    if (cudaSuccess != cudaMemcpy(h_ptr, d_ptr, nbytes, cudaMemcpyDeviceToHost)) {
        throw HostDeviceCopyFailedError("D2H");
    }
#elif defined(HPGMP_WITH_HIP)
    if (hipSuccess != hipMemcpy(h_ptr, d_ptr, nbytes, hipMemcpyDeviceToHost)) {
        throw HostDeviceCopyFailedError("D2H");
    }
#endif
}

void DeviceCtx::copy_device_to_device_sync(void* dst_ptr, const void* src_ptr, const size_t nbytes)
{
#if defined(HPGMP_WITH_CUDA)
    if (cudaSuccess != cudaMemcpy(dst_ptr, src_ptr, nbytes, cudaMemcpyDeviceToDevice)) {
        throw HostDeviceCopyFailedError("D2D");
    }
#elif defined(HPGMP_WITH_HIP)
    if (hipSuccess != hipMemcpy(dst_ptr, src_ptr, nbytes, hipMemcpyDeviceToDevice)) {
        throw HostDeviceCopyFailedError("D2D");
    }
#endif
}

void DeviceCtx::copy_host_to_device_async(void* d_ptr, const void* h_ptr, size_t nbytes,
                                          stream_t stream)
{
#if defined(HPGMP_WITH_CUDA)
    if (cudaSuccess != cudaMemcpyAsync(d_ptr, h_ptr, nbytes, cudaMemcpyHostToDevice, stream)) {
        throw HostDeviceCopyFailedError("H2D");
    }
#elif defined(HPGMP_WITH_HIP)
    if (hipSuccess != hipMemcpyAsync(d_ptr, h_ptr, nbytes, hipMemcpyHostToDevice, stream)) {
        throw HostDeviceCopyFailedError("H2D");
    }
#endif
}

void DeviceCtx::copy_device_to_host_async(void* h_ptr, const void* d_ptr, size_t nbytes,
                                          stream_t stream)
{
#if defined(HPGMP_WITH_CUDA)
    if (cudaSuccess != cudaMemcpyAsync(h_ptr, d_ptr, nbytes, cudaMemcpyDeviceToHost, stream)) {
        throw HostDeviceCopyFailedError("D2H");
    }
#elif defined(HPGMP_WITH_HIP)
    if (hipSuccess != hipMemcpyAsync(h_ptr, d_ptr, nbytes, hipMemcpyDeviceToHost, stream)) {
        throw HostDeviceCopyFailedError("D2H");
    }
#endif
}

void DeviceCtx::synchronize_compute_stream()
{
#if defined(HPGMP_WITH_CUDA)
    cudaStreamSynchronize(compute_stream_);
#elif defined(HPGMP_WITH_HIP)
    HPGMP_THROW_ON_ERROR(hipStreamSynchronize(compute_stream_), "sync compute stream");
#endif
}

void DeviceCtx::synchronize_halo_stream()
{
#if defined(HPGMP_WITH_CUDA)
    cudaStreamSynchronize(halo_stream_);
#elif defined(HPGMP_WITH_HIP)
    HPGMP_THROW_ON_ERROR(hipStreamSynchronize(halo_stream_), "sync halo stream");
#endif
}

void DeviceCtx::synchronize_device()
{
#if defined(HPGMP_WITH_CUDA)
    cudaDeviceSynchronize();
#elif defined(HPGMP_WITH_HIP)
    HPGMP_THROW_ON_ERROR(hipDeviceSynchronize(), "sync device");
#endif
}

event_t DeviceCtx::create_event()
{
    event_t event;
#if defined(HPGMP_WITH_CUDA)
    if (cudaSuccess != cudaEventCreate(&event)) {
        throw DeviceAPIError("Could not create event!");
    }
#elif defined(HPGMP_WITH_HIP)
    if (hipSuccess != hipEventCreate(&event)) {
        throw DeviceAPIError("Could not create event!");
    }
#endif
    return event;
}

void DeviceCtx::destroy_event(event_t event)
{
#if defined(HPGMP_WITH_CUDA)
    if (cudaSuccess != cudaEventDestroy(event)) {
        throw DeviceAPIError("Could not create event!");
    }
#elif defined(HPGMP_WITH_HIP)
    if (hipSuccess != hipEventDestroy(event)) {
        throw DeviceAPIError("Could not create event!");
    }
#endif
}

void DeviceCtx::record_event(event_t event, stream_t stream)
{
#if defined(HPGMP_WITH_CUDA)
    if (cudaSuccess != cudaEventRecord(event, stream)) {
        throw DeviceAPIError("Could not record event!");
    }
#elif defined(HPGMP_WITH_HIP)
    if (hipSuccess != hipEventRecord(event, stream)) {
        throw DeviceAPIError("Could not record event!");
    }
#else
#error "Events not supported on host yet!"
#endif
}

void DeviceCtx::stream_wait_on_event(stream_t stream, event_t event)
{
#if defined(HPGMP_WITH_CUDA)
    if (cudaSuccess != cudaStreamWaitEvent(stream, event, 0)) {
        throw DeviceAPIError("Could not record event!");
    }
#elif defined(HPGMP_WITH_HIP)
    if (hipSuccess != hipStreamWaitEvent(stream, event, 0)) {
        throw DeviceAPIError("Could not record event!");
    }
#else
#error "Events not supported on host yet!"
#endif
}

void DeviceCtx::device_memset(void* d_ptr, int value, size_t nbytes)
{
#if defined(HPGMP_WITH_CUDA)
    if (cudaSuccess != cudaMemset(d_ptr, value, nbytes)) {
        throw DeviceAPIError("memset");
    }
#elif defined(HPGMP_WITH_HIP)
    if (hipSuccess != hipMemset(d_ptr, value, nbytes)) {
        throw DeviceAPIError("memset");
    }
#endif
}

#ifdef HPGMP_WITH_HIP
const std::string DeviceAPIError::platform = "HIP";
#elif HPGMP_WITH_CUDA
const std::string DeviceAPIError::platform = "CUDA";
#else
const std::string DeviceAPIError::platform = "CPU";
#endif
