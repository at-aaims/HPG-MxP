
//@HEADER
// ***************************************************
//
// HPGMP: High Performance Generalized minimal residual
//        - Mixed-Precision
//
// Contact:
// Ichitaro Yamazaki         (iyamaza@sandia.gov)
// Sivasankaran Rajamanickam (srajama@sandia.gov)
// Piotr Luszczek            (luszczek@eecs.utk.edu)
// Jack Dongarra             (dongarra@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file ComputeGS_Forward_ref.cpp

 HPGMP routine
 */
#if !defined(HPGMP_WITH_CUDA) & !defined(HPGMP_WITH_HIP)

#ifndef HPGMP_NO_MPI
 #include "ExchangeHalo.hpp"
#endif
#include "ComputeGS_Forward_ref.hpp"
#include "mytimer.hpp"
#include <cassert>
#include <iostream>

/*!
  Computes one forward step of Gauss-Seidel:

  Assumption about the structure of matrix A:
  - Each row 'i' of the matrix has nonzero diagonal value whose address is matrixDiagonal[i]
  - Entries in row 'i' are ordered such that:
       - lower triangular terms are stored before the diagonal element.
       - upper triangular terms are stored after the diagonal element.
       - No other assumptions are made about entry ordering.

  Gauss-Seidel notes:
  - We use the input vector x as the RHS and start with an initial guess for y of all zeros.
  - We perform one forward sweep.  x should be initially zero on the first GS sweep, but we do not attempt to exploit this fact.

  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On entry, x should contain relevant values, on exit x contains the result of one symmetric GS sweep with r as the RHS.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeGS_Forward
*/
template<class SparseMatrix_type, class Vector_type>
int ComputeGS_Forward_ref(const SparseMatrix_type & A, const Vector_type & r, Vector_type & x) {

  assert(x.localLength==A.localNumberOfColumns); // Make sure x contain space for halo values

  typedef typename SparseMatrix_type::scalar_type scalar_type;
  const local_int_t nrow = A.localNumberOfRows;

  const scalar_type * const rv = r.values;
  scalar_type * const xv = x.values;

  double t0 = 0.0;
#ifndef HPGMP_NO_MPI
  // Exchange Halo on HOST CPU
  ExchangeHalo(A, x);
#endif

  scalar_type ** matrixDiagonal = A.matrixDiagonal;  // An array of pointers to the diagonal entries A.matrixValues

  TICK();
  for (local_int_t i=0; i < nrow; i++) {
    const scalar_type * const currentValues = A.matrixValues[i];
    const local_int_t * const currentColIndices = A.mtxIndL[i];
    const int currentNumberOfNonzeros = A.nonzerosInRow[i];
    const scalar_type currentDiagonal = matrixDiagonal[i][0]; // Current diagonal value
    scalar_type sum = rv[i]; // RHS value

    for (int j=0; j< currentNumberOfNonzeros; j++) {
      local_int_t curCol = currentColIndices[j];
      sum -= currentValues[j] * xv[curCol];
    }
    sum += xv[i]*currentDiagonal; // Remove diagonal contribution from previous loop

    xv[i] = sum/currentDiagonal;
  }
  TOCK(x.time2);

  return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

template
int ComputeGS_Forward_ref< SparseMatrix<double>, Vector<double> >(SparseMatrix<double> const&, Vector<double> const&, Vector<double>&);

template
int ComputeGS_Forward_ref< SparseMatrix<float>, Vector<float> >(SparseMatrix<float> const&, Vector<float> const&, Vector<float>&);

#endif
