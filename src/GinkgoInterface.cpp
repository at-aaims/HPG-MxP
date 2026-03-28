#ifdef HPGMP_WITH_GINKGO
#include "GinkgoInterface.hpp"

std::shared_ptr<gko::Executor> create_ginkgo_executor()
{
#if defined(HPGMP_WITH_HIP) || defined(HPGMP_WITH_CUDA)
    return gko_exec_type::create(0, gko_reference_exec_type::create());
#else
    return gko_exec_type::create();
#endif
}

#endif
