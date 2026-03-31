#ifndef HPGMP_RESTRICTION_HPP
#define HPGMP_RESTRICTION_HPP

#include "SparseMatrix.hpp"
#include "Vector.hpp"

template<typename mat_scalar_type, typename vec_scalar_type>
int fused_spmv_restriction(const SparseMatrix<mat_scalar_type>& A, const Vector<vec_scalar_type>& rf,
                           const Vector<vec_scalar_type>& xf);

template<typename mat_scalar_type, typename vec_scalar_type>
int restriction(const SparseMatrix<mat_scalar_type>& A, const Vector<vec_scalar_type>& rf);

#endif
