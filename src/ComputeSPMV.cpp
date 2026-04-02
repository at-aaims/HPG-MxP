
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
 @file ComputeSPMV.cpp

 HPGMP routine
 */

#include "ComputeSPMV.hpp"
#ifdef HPGMP_REFERENCE
#include "ComputeSPMV_ref.hpp"
#else // HPGMP_REFERENCE
#ifdef HPGMP_WITH_GINKGO
#include "GinkgoMatrix.hpp"
#else
#include "ell_matrix.hpp"
#endif
#endif // HPGMP_REFERENCE

/*!
  Routine to compute sparse matrix vector product y = Ax where:
  Precondition: First call exchange_externals to get off-processor values of x

  This routine calls the reference SpMV implementation by default, but
  can be replaced by a custom, optimized routine suited for
  the target system.

  @param[in]  A the known system matrix
  @param[in]  x the known vector
  @param[out] y the On exit contains the result: Ax.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeSPMV_ref
*/
template<class SparseMatrix_type, class Vector_type>
int ComputeSPMV(const SparseMatrix_type& A, Vector_type& x, Vector_type& y)
{
#ifdef HPGMP_REFERENCE
    // This line and the next two lines should be removed and your version of ComputeSPMV should be used.
    A.isSpmvOptimized = false;
    return ComputeSPMV_ref(A, x, y);
#else
    A.isSpmvOptimized = true;
#ifdef HPGMP_WITH_GINKGO
    return ComputeSPMV_ginkgo(A, x, y);
#else
    return ComputeSPMV_ell(A, x, y);
#endif
#endif
}


/* --------------- *
 * specializations *
 * --------------- */

template int ComputeSPMV< SparseMatrix<double>, Vector<double> >(
    const SparseMatrix<double>&, Vector<double>&, Vector<double>&);

template int ComputeSPMV< SparseMatrix<float>, Vector<float> >(
    const SparseMatrix<float>&, Vector<float>&, Vector<float>&);

template int ComputeSPMV< SparseMatrix<double, float>, Vector<float> >(
    const SparseMatrix<double, float>&, Vector<float>&, Vector<float>&);

template int ComputeSPMV< SparseMatrix<double, float>, Vector<double> >(
    const SparseMatrix<double, float>&, Vector<double>&, Vector<double>&);
