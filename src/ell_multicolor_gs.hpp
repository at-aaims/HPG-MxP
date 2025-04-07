#ifndef HPGMP_ELL_MULTICOLOR_GS_HPP
#define HPGMP_ELL_MULTICOLOR_GS_HPP

#include "ell_matrix.hpp"
#include "Vector.hpp"

/*!
 * @brief Routine to compute one step of forward Gauss-Seidel.
 *
 * Assumption about the structure of matrix A:
 * - Each row 'i' of the matrix has nonzero diagonal value whose address is matrixDiagonal[i]
 * - Entries in row 'i' are ordered such that:
 *      - lower triangular terms are stored before the diagonal element.
 *      - upper triangular terms are stored after the diagonal element.
 *      - No other assumptions are made about entry ordering.
 *
 * Symmetric Gauss-Seidel notes:
 * - We use the input vector x as the RHS and start with an initial guess for y of all zeros.
 * - We perform one forward sweep.  Since y is initially zero we can ignore the upper triangular terms of A.
 * - We then perform one back sweep.
 *      - For simplicity we include the diagonal contribution in the for-j loop, then correct the sum after
 *
 * @param[in] symmetric  Whether to apply symmetric Gauss-Seidel (true) or not (false)
 * @param[in] A the known system matrix
 * @param[in] r the input vector
 * @param[inout] x On entry, x should contain relevant values, on exit x contains
 *                 the result of one GS sweep with r as the RHS.
 *
 * @return returns 0 upon success and non-zero otherwise
 */
template <typename mscalar, typename vscalar>
int ell_multicolor_gs(bool symmetric, const ELLMatrix<mscalar> *A, const Vector<vscalar> *r,
                     Vector<vscalar> *x);

template <typename mscalar, typename vscalar>
int ell_multicolor_gs_zero_initial(bool symmetric, const ELLMatrix<mscalar> *A,
        const Vector<vscalar> *r, Vector<vscalar> *x);

#endif
