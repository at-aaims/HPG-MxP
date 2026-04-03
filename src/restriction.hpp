#ifndef HPGMP_RESTRICTION_HPP
#define HPGMP_RESTRICTION_HPP

#include "SparseMatrix.hpp"
#include "Vector.hpp"

template<typename local_scalar_t, typename halo_scalar_t, typename vec_scalar_t>
int fused_spmv_restriction(const SparseMatrix<local_scalar_t, halo_scalar_t>& A, const Vector<vec_scalar_t>& rf,
                           const Vector<vec_scalar_t>& xf);

template<typename local_scalar_t, typename halo_scalar_t, typename vec_scalar_t>
int restriction(const SparseMatrix<local_scalar_t, halo_scalar_t>& A, const Vector<vec_scalar_t>& rf);

#endif
