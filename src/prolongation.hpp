#ifndef HPGMP_PROLONGATION_HPP
#define HPGMP_PROLONGATION_HPP

#include "Vector.hpp"
#include "SparseMatrix.hpp"

template<typename local_scalar_t, typename halo_scalar_t, typename vec_scalar_t>
int prolongation(const SparseMatrix<local_scalar_t, halo_scalar_t>& Af, Vector<vec_scalar_t>& xf);

#endif
