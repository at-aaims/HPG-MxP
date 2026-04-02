#ifndef HPGMP_RESTRICTION_HPP
#define HPGMP_RESTRICTION_HPP

#include "SparseMatrix.hpp"
#include "Vector.hpp"

template<typename local_scalar_t, typename halo_scalar_t, typename vec_scalar_type>
int fused_spmv_restriction(const SparseMatrix<local_scalar_t, halo_scalar_t>& A, const Vector<vec_scalar_type>& rf,
                           const Vector<vec_scalar_type>& xf);

template<typename local_scalar_t, typename halo_scalar_t, typename vec_scalar_type>
int restriction(const SparseMatrix<local_scalar_t, halo_scalar_t>& A, const Vector<vec_scalar_type>& rf);

#endif
