#ifndef HPGMP_GINKGO_INTERFACE
#define HPGMP_GINKGO_INTERFACE

#include "ell_matrix.hpp"
#include "Profiling.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"

#include <ginkgo/ginkgo.hpp>

#ifdef HPGMP_WITH_HIP
auto ginkgo_exec = gko::HipExecutor::create(0, gko::ReferenceExecutor::create());
#elif HPGMP_WITH_CUDA
auto ginkgo_exec = gko::CudaExecutor::create(0, gko::ReferenceExecutor::create());
#else // CPU
#ifdef HPGMP_NO_OPENMP
auto ginkgo_exec = gko::ReferenceExecutor::create();
#else // OPENMP
auto ginkgo_exec = gko::OmpExecutor::create();
#endif // HPGMP_NO_OPENMP
#endif // HPGMP_WITH_HIP of HPGMP_WITH_CUDA

template<class SparseMatrix_type, class Vector_type>
int GinkgoTest(const SparseMatrix_type& A, const Vector_type& r, Vector_type& x)
{

    HPGMP_RANGE_PUSH(__FUNCTION__);

    HPGMP_RANGE_POP(__FUNCTION__);

    return 0;
}

#endif // HPGMP_GINKGO_INTERFACE
