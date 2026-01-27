
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
 @file ComputeProlongation_ref.cpp

 HPGMP routine
 */
#if !defined(HPGMP_WITH_CUDA) & !defined(HPGMP_WITH_HIP)

#ifndef HPGMP_NO_OPENMP
#include <omp.h>
#endif

#include "ComputeProlongation_ref.hpp"

/*!
  Routine to compute the coarse residual vector.

  @param[in]  Af - Fine grid sparse matrix object containing pointers to current coarse grid correction and the f2c operator.
  @param[inout] xf - Fine grid solution vector, update with coarse grid correction.

  Note that the fine grid residual is never explicitly constructed.
  We only compute it for the fine grid points that will be injected into corresponding coarse grid points.

  @return Returns zero on success and a non-zero value otherwise.
*/
template<class SparseMatrix_type, class Vector_type>
int ComputeProlongation_ref(const SparseMatrix_type& Af, Vector_type& xf)
{

    typedef typename SparseMatrix_type::scalar_type scalar_type;

    scalar_type* xfv       = xf.values();
    const scalar_type* xcv = Af.mgData->xc->values();
    local_int_t* f2c       = Af.mgData->f2cOperator;
    local_int_t nc         = Af.mgData->rc->local_length();

#ifndef HPGMP_NO_OPENMP
    // clang-format off
    #pragma omp parallel for
    // clang-format on
#endif
    // TODO: Somehow note that this loop can be safely vectorized since f2c has no repeated indices
    for (local_int_t i = 0; i < nc; ++i) xfv[f2c[i]] += xcv[i]; // This loop is safe to vectorize

    return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

template int ComputeProlongation_ref< SparseMatrix<double>, Vector<double> >(
    SparseMatrix<double> const&, Vector<double>&);

template int ComputeProlongation_ref< SparseMatrix<float>, Vector<float> >(
    SparseMatrix<float> const&, Vector<float>&);

#endif
