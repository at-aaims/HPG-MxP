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

#if defined(HPGMP_WITH_CUDA) || defined(HPGMP_WITH_HIP)

#include "random_gpu.hpp"

#include <stdexcept>
#include <type_traits>

#ifdef HPGMP_WITH_HIP
#include <rocrand/rocrand.hpp>
#elif defined HPGMP_WITH_CUDA
#include <curand.h>
#endif

namespace {

    template<typename scalar>
    void fill_random_gpu_impl(scalar* const x, const size_t n, random_stream_t stream)
    {
#ifdef HPGMP_WITH_HIP
        rocrand_generator generator;
        rocrand_create_generator(&generator, ROCRAND_RNG_PSEUDO_DEFAULT);
        rocrand_set_stream(generator, stream);
        rocrand_initialize_generator(generator);
        //const rocblas_datatype alpha_type = rocblas_datatype_f64_r;
        //rocblas_datatype x_type;
        //auto handle = dctx_->get_blas_handle();
        rocrand_status status = ROCRAND_STATUS_SUCCESS;
        if constexpr (std::is_same<scalar, double>::value) {
            status = rocrand_generate_uniform_double(generator, x, n);
            //x_type = rocblas_datatype_f_64_r;
        } else {
            status = rocrand_generate_uniform(generator, x, n);
            //x_type = rocblas_datatype_f_32_r;
        }
        if (status != ROCRAND_STATUS_SUCCESS) {
            throw std::runtime_error("random vector generation failed!");
        }
        if (hipSuccess != hipStreamSynchronize(stream)) {
            throw std::runtime_error("random vector generation: stream synchronization failed!");
        }
        // TODO: Add 1 to generated vector
        rocrand_destroy_generator(generator);
#else
        curandGenerator_t generator;
        curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetStream(generator, stream);
        curandStatus status = CURAND_STATUS_SUCCESS;
        if constexpr (std::is_same<scalar, double>::value) {
            status = curandGenerateUniformDouble(generator, x, n);
        } else {
            status = curandGenerateUniform(generator, x, n);
        }
        if (status != CURAND_STATUS_SUCCESS) {
            throw std::runtime_error("random vector generation failed!");
        }
        if (cudaSuccess != cudaStreamSynchronize(stream)) {
            throw std::runtime_error("random vector generation: stream synchronization failed!");
        }
        // TODO: Add 1 to generated vector
        curandDestroyGenerator(generator);
#endif
    }

} // namespace

void fill_random_gpu(double* const x, const size_t n, random_stream_t stream)
{
    fill_random_gpu_impl(x, n, stream);
}

void fill_random_gpu(float* const x, const size_t n, random_stream_t stream)
{
    fill_random_gpu_impl(x, n, stream);
}

#endif
