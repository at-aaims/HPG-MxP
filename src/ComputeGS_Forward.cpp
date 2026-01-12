
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
 @file ComputeSYMGS.cpp

 HPGMP routine
 */

#include "ComputeGS_Forward.hpp"
#include "ComputeGS_Forward_ref.hpp"

/*!
  Routine to compute one forward step of Gauss-Seidel:

  Assumption about the structure of matrix A:
  - Each row 'i' of the matrix has nonzero diagonal value whose address is matrixDiagonal[i]
  - Entries in row 'i' are ordered such that:
       - lower triangular terms are stored before the diagonal element.
       - upper triangular terms are stored after the diagonal element.
       - No other assumptions are made about entry ordering.

  Gauss-Seidel notes:
  - We use the input vector x as the RHS and start with an initial guess for y of all zeros.
  - We perform one forward sweep.  Since y is initially zero we can ignore the upper triangular terms of A.

  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On entry, x should contain relevant values, on exit x contains the result of one symmetric GS sweep with r as the RHS.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeGS_Forward_ref
*/
template<class SparseMatrix_type, class Vector_type>
int ComputeGS_Forward(const SparseMatrix_type& A, const Vector_type& r, Vector_type& x)
{

    // This line and the next two lines should be removed and your version of ComputeSYMGS should be used.
    return ComputeGS_Forward_ref(A, r, x);
}

template int ComputeGS_Forward< SparseMatrix<double>, Vector<double> >(
    SparseMatrix<double> const&, Vector<double> const&, Vector<double>&);

template int ComputeGS_Forward< SparseMatrix<float>, Vector<float> >(
    SparseMatrix<float> const&, Vector<float> const&, Vector<float>&);
