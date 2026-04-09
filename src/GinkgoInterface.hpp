#ifndef HPGMP_GINKGO_INTERFACE_HPP
#define HPGMP_GINKGO_INTERFACE_HPP

#include <ginkgo/ginkgo.hpp>
#include "device_ctx.hpp"

using gko_reference_exec_type = gko::ReferenceExecutor;
#ifdef HPGMP_WITH_HIP
using gko_exec_type      = gko::HipExecutor;
using gko_allocator_type = gko::HipAllocator;
#elif HPGMP_WITH_CUDA
using gko_exec_type      = gko::CudaExecutor;
using gko_allocator_type = gko::CudaAllocator;
#else // CPU
#ifdef HPGMP_NO_OPENMP
using gko_exec_type = gko_reference_exec_type;
#else // OPENMP
using gko_exec_type = gko::OmpExecutor;
#endif // HPGMP_NO_OPENMP
#endif // HPGMP_WITH_HIP or HPGMP_WITH_CUDA

std::shared_ptr<gko::Executor> create_ginkgo_executor(stream_t stream);

#endif // HPGMP_GINKGO_INTERFACE
