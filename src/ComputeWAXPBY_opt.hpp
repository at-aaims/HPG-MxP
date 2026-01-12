#ifndef HPGMP_COMPUTE_WAXPBY_OPT_HPP
#define HPGMP_COMPUTE_WAXPBY_OPT_HPP

template<class VectorX_type, class VectorY_type, class VectorW_type>
int ComputeWAXPBY_opt(local_int_t n,
                      typename VectorX_type::scalar_type alpha,
                      const VectorX_type& x,
                      typename VectorY_type::scalar_type beta,
                      const VectorY_type& y,
                      VectorW_type& w,
                      bool& isoptimized);

#endif
