#ifndef HPGMP_COMPUTE_WAXPBY_OPT_HPP
#define HPGMP_COMPUTE_WAXPBY_OPT_HPP

// alpha and beta are passed by reference to support half with HIP.
template<class VectorX_type, class VectorY_type, class VectorW_type>
int ComputeWAXPBY_opt(local_int_t n,
                      const typename VectorX_type::scalar_type& alpha,
                      const VectorX_type& x,
                      const typename VectorY_type::scalar_type& beta,
                      const VectorY_type& y,
                      VectorW_type& w);

#endif
