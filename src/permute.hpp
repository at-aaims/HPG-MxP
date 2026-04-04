#ifndef HPGMP_PERMUTE_HPP
#define HPGMP_PERMUTE_HPP

#include "SparseMatrix.hpp"
#include "ell_matrix.hpp"

template<typename local_scalar_t, typename halo_scalar_t>
void permute_columns(SparseMatrix<local_scalar_t, halo_scalar_t>& A);

#endif
