#ifndef HPGMP_MULTICOLORING_HPP
#define HPGMP_MULTICOLORING_HPP

#include "SparseMatrix.hpp"

/** @brief Parallel independent set algorithm for GPU.
 *
 * Computes a permutation vector perm of mesh points, such that
 * point_new_index = A.perm[point_old_index].
 */
template <typename scalar>
void multicolor_JPL(SparseMatrix<scalar>& A);

/** @brief Sequential greedy independent set algorithm on CPU.
 *
 * Computes a permutation vector perm of mesh points, such that
 * point_new_index = A.perm[point_old_index].
 */
template <typename scalar>
void multicolor_ref(SparseMatrix<scalar>& A);

#endif
