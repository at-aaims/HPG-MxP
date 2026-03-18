#ifndef HPGMP_GINKGO_INTERFACE
#define HPGMP_GINKGO_INTERFACE

#include "ell_matrix.hpp"
#include "Profiling.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"

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
#endif // HPGMP_WITH_HIP of HPGMP_WITH_CUDA

std::shared_ptr<gko::Executor> create_ginkgo_executor()
{
#if defined(HPGMP_WITH_HIP) || defined(HPGMP_WITH_CUDA)
    return gko_exec_type::create(0, gko_reference_exec_type::create());
#else
    return gko_exec_type::create();
#endif
}

#endif // HPGMP_GINKGO_INTERFACE
