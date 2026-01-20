
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
 @file ComputeRestriction_ref.cpp

 HPGMP routine
 */
#if !defined(HPGMP_WITH_CUDA) & !defined(HPGMP_WITH_HIP)

#ifndef HPGMP_NO_OPENMP
#include <omp.h>
#endif

#include "ComputeRestriction_ref.hpp"

/*!
  Routine to compute the coarse residual vector.

  @param[inout]  A - Sparse matrix object containing pointers to mgData->Axf, the fine grid matrix-vector product and mgData->rc the coarse residual vector.
  @param[in]    rf - Fine grid RHS.


  Note that the fine grid residual is never explicitly constructed.
  We only compute it for the fine grid points that will be injected into corresponding coarse grid points.

  @return Returns zero on success and a non-zero value otherwise.
*/
template<class SparseMatrix_type, class Vector_type>
int ComputeRestriction_ref(const SparseMatrix_type& A, const Vector_type& rf)
{

    typedef typename SparseMatrix_type::scalar_type scalar_type;

    scalar_type* Axfv = A.mgData->Axf->values;
    scalar_type* rfv  = rf.values;
    scalar_type* rcv  = A.mgData->rc->values;
    local_int_t* f2c  = A.mgData->f2cOperator;
    local_int_t nc    = A.mgData->rc->localLength;

#ifndef HPGMP_NO_OPENMP
    // clang-format off
    #pragma omp parallel for
    // clang-format on
#endif
    for (local_int_t i = 0; i < nc; ++i) rcv[i] = rfv[f2c[i]] - Axfv[f2c[i]];

    return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

template int ComputeRestriction_ref< SparseMatrix<double>, Vector<double> >(
    SparseMatrix<double> const&, Vector<double> const&);

template int ComputeRestriction_ref< SparseMatrix<float>, Vector<float> >(
    SparseMatrix<float> const&, Vector<float> const&);

#endif
