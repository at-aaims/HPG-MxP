
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
 @file WriteProblem.cpp

 HPGMP routine
 */

#include <cstdio>
#include "WriteProblem.hpp"


/*!
  Routine to dump:
   - matrix in row, col, val format for analysis with MATLAB
   - x, xexact, b as simple arrays of numbers.

   Writes to A.dat, x.dat, xexact.dat and b.dat, respectivly.

   NOTE:  THIS CODE ONLY WORKS ON SINGLE PROCESSOR RUNS

   Read into MATLAB using:

       load A.dat
       A=spconvert(A);
       load x.dat
       load xexact.dat
       load b.dat

  @param[in] geom   The description of the problem's geometry.
  @param[in] A      The known system matrix
  @param[in] b      The known right hand side vector
  @param[in] x      The solution vector computed by CG iteration
  @param[in] xexact Generated exact solution

  @return Returns with -1 if used with more than one MPI process. Returns with 0 otherwise.

  @see GenerateProblem
*/
template<class SparseMatrix_type, class Vector_type>
int WriteProblem(const Geometry& geom, const SparseMatrix_type& A,
                 const Vector_type b, const Vector_type x, const Vector_type xexact)
{

    typedef typename SparseMatrix_type::scalar_type scalar_type;
    if (geom.size != 1) return -1; //TODO Only works on one processor.  Need better error handler
    const global_int_t nrow = A.totalNumberOfRows;

    FILE *fA = 0, *fx = 0, *fxexact = 0, *fb = 0;
    fA      = fopen("A.dat", "w");
    fx      = fopen("x.dat", "w");
    fxexact = fopen("xexact.dat", "w");
    fb      = fopen("b.dat", "w");

    if (!fA || !fx || !fxexact || !fb) {
        if (fb) fclose(fb);
        if (fxexact) fclose(fxexact);
        if (fx) fclose(fx);
        if (fA) fclose(fA);
        return -1;
    }

    for (global_int_t i = 0; i < nrow; i++) {
        const scalar_type* const currentRowValues   = A.matrixValues[i];
        const global_int_t* const currentRowIndices = A.mtxIndG[i];
        const int currentNumberOfNonzeros           = A.nonzerosInRow[i];
        for (int j = 0; j < currentNumberOfNonzeros; j++)
#ifdef HPGMP_NO_LONG_LONG
            fprintf(fA, " %d %d %22.16e\n", i + 1, (global_int_t)(currentRowIndices[j] + 1), currentRowValues[j]);
#else
            fprintf(fA, " %lld %lld %22.16e\n", i + 1, (global_int_t)(currentRowIndices[j] + 1), currentRowValues[j]);
#endif
        fprintf(fx, "%22.16e\n", x.values[i]);
        fprintf(fxexact, "%22.16e\n", xexact.values[i]);
        fprintf(fb, "%22.16e\n", b.values[i]);
    }

    fclose(fA);
    fclose(fx);
    fclose(fxexact);
    fclose(fb);
    return 0;
}
