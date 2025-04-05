#ifndef HPGMP_PROLONGATION_HPP
#define HPGMP_PROLONGATION_HPP

#include "Vector.hpp"
#include "SparseMatrix.hpp"

template <typename mat_scalar, typename vec_scalar>
int prolongation(const SparseMatrix<mat_scalar>& Af, Vector<vec_scalar>& xf);

#endif
