
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
 @file SetupHalo.cpp

 HPGMP routine
 */

#ifndef HPGMP_NO_MPI
#include <mpi.h>
#include <map>
#include <set>
#endif

#ifndef HPGMP_NO_OPENMP
#include <omp.h>
#endif

#include "SetupHalo.hpp"
#include "SetupHalo_ref.hpp"

/*!
  Prepares system matrix data structure and creates data necessary necessary
  for communication of boundary values of this process.

  @param[inout] A    The known system matrix

  @see ExchangeHalo
*/
template<class SparseMatrix_type>
void SetupHalo(SparseMatrix_type& A)
{

    // The call to this reference version of SetupHalo can be replaced with custom code.
    // However, any code must work for general unstructured sparse matrices.  Special knowledge about the
    // specific nature of the sparsity pattern may not be explicitly used.

    return (SetupHalo_ref(A));
}

/* --------------- *
 * specializations *
 * --------------- */

template void SetupHalo< SparseMatrix<double> >(SparseMatrix<double>&);
template void SetupHalo< SparseMatrix<float> >(SparseMatrix<float>&);
template void SetupHalo< SparseMatrix<double, float> >(SparseMatrix<double, float>&);
