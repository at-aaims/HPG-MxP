#ifndef HPGMP_GINKGO_INTERFACE
#define HPGMP_GINKGO_INTERFACE

#include <ginkgo/ginkgo.hpp>

using gko_reference_exec_type = gko::ReferenceExecutor;
#ifdef HPGMP_WITH_HIP
using gko_exec_type = gko::HipExecutor;
#elif HPGMP_WITH_CUDA
using gko_exec_type = gko::CudaExecutor;
#else // CPU
#ifdef HPGMP_NO_OPENMP
using gko_exec_type = gko_reference_exec_type;
#else // OPENMP
using gko_exec_type = gko::OmpExecutor;
#endif // HPGMP_NO_OPENMP
#endif // HPGMP_WITH_HIP or HPGMP_WITH_CUDA

std::shared_ptr<gko::Executor> create_ginkgo_executor();

#endif // HPGMP_GINKGO_INTERFACE
