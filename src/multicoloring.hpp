#ifndef HPGMP_MULTICOLORING_HPP
#define HPGMP_MULTICOLORING_HPP

#include "SparseMatrix.hpp"

/** @brief Parallel independent set algorithm for GPU.
 *
 * Computes a permutation vector perm of mesh points, such that
 * point_new_index = A.perm[point_old_index].
 */
template<typename local_scalar_t, typename halo_scalar_t>
void multicolor_JPL(SparseMatrix<local_scalar_t, halo_scalar_t>& A);

/** @brief Sequential greedy independent set algorithm on CPU.
 *
 * Computes a permutation vector perm of mesh points, such that
 * point_new_index = A.perm[point_old_index].
 */
template<typename local_scalar_t, typename halo_scalar_t>
void multicolor_ref(SparseMatrix<local_scalar_t, halo_scalar_t>& A);

#endif
