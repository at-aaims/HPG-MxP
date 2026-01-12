
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
 @file ExchangeHalo.cpp

 HPGMP routine
 */

// Compile this routine only if running with MPI
#ifndef HPGMP_NO_MPI
#include "ExchangeHalo.hpp"
#include "ExchangeHalo_ref.hpp"

/*!
  Communicates data that is at the border of the part of the domain assigned to this processor.

  @param[in]    A The known system matrix
  @param[inout] x On entry: the local vector entries followed by entries to be communicated; on exit: the vector with non-local entries updated by other processors
 */
template<class SparseMatrix_type, class Vector_type>
void ExchangeHalo(const SparseMatrix_type& A, Vector_type& x)
{

    ExchangeHalo_ref(A, x);

    return;
}


/* --------------- *
 * specializations *
 * --------------- */

template void ExchangeHalo< SparseMatrix<double>, Vector<double> >(
    SparseMatrix<double> const&, Vector<double>&);

template void ExchangeHalo< SparseMatrix<float>, Vector<float> >(
    SparseMatrix<float> const&, Vector<float>&);

#endif // ifndef HPGMP_NO_MPI
