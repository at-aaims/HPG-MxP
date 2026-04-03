#ifdef HPGMP_WITH_GINKGO
#include "GinkgoOptData.hpp"

// Available instantiations
template class GinkgoOptData<double, double>;
template class GinkgoOptData<float, float>;
template class GinkgoOptData<double, float>;

#endif
