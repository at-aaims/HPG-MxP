#ifdef HPGMP_WITH_GINKGO

#include "GinkgoInterface.hpp"

std::shared_ptr<gko::Executor> create_ginkgo_executor(stream_t stream)
{
#if defined(HPGMP_WITH_HIP) || defined(HPGMP_WITH_CUDA)
    auto allocator = std::make_shared<gko_allocator_type>();
    return gko_exec_type::create(0, gko_reference_exec_type::create(), allocator, stream);
#else
    return gko_exec_type::create();
#endif
}

#endif
