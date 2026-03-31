#ifndef HPGMP_PROLONGATION_HPP
#define HPGMP_PROLONGATION_HPP

#include "Vector.hpp"
#include "SparseMatrix.hpp"

template<typename mat_scalar_type, typename vec_scalar_type>
int prolongation(const SparseMatrix<mat_scalar_type>& Af, Vector<vec_scalar_type>& xf);

#endif
