#ifndef HPGMP_RESTRICTION_HPP
#define HPGMP_RESTRICTION_HPP

#include "SparseMatrix.hpp"
#include "Vector.hpp"

template <typename mscalar, typename vscalar>
int fused_spmv_restriction(const SparseMatrix<mscalar>& A, const Vector<vscalar>& rf,
                           const Vector<vscalar>& xf);

template <typename mscalar, typename vscalar>
int restriction(const SparseMatrix<mscalar>& A, const Vector<vscalar>& rf);

#endif
