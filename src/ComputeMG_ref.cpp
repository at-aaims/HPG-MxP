
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
 @file ComputeSYMGS_ref.cpp

 HPGMP routine
 */

#include "ComputeMG_ref.hpp"
#include "ComputeSYMGS_ref.hpp"
#include "ComputeGS_Forward_ref.hpp"
#include "ComputeSPMV_ref.hpp"
#include "ComputeRestriction_ref.hpp"
#include "ComputeProlongation_ref.hpp"
#ifdef HPGMP_DEBUG
#include "hpgmp.hpp"
#endif
#include "mytimer.hpp"
#include <cassert>
#include <iostream>

/*!

  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On exit contains the result of the multigrid V-cycle with r as the RHS, x is the approximation to Ax = r.

  @return returns 0 upon success and non-zero otherwise

  The total number of flops in MG with N levels is 2*nnz_allgrids*(1 + n_presmooth + n_postsmooth).

  @see ComputeMG
*/
template<class SparseMatrix_type, class Vector_type>
int ComputeMG(const SparseMatrix_type& A, const Vector_type& r, Vector_type& x, bool symmetric, perf_counters& ft)
{

    HPGMP_RANGE_PUSH(__FUNCTION__);

    assert(x.local_length() == A.localNumberOfColumns); // Make sure x contain space for halo values

    // initialize x to zero
    double t0 = 0.0;
    x.fill_zero();

    int ierr = 0;
    if (A.mgData != 0) { // Go to next coarse level if defined
        const int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;
        if (symmetric) {
            for (int i = 0; i < numberOfPresmootherSteps; ++i)
                ierr += ComputeSYMGS_ref(A, r, x);
        } else {
            for (int i = 0; i < numberOfPresmootherSteps; ++i)
                ierr += ComputeGS_Forward_ref(A, r, x);
        }
        if (ierr != 0) {
            HPGMP_RANGE_POP(__FUNCTION__);
            return ierr;
        }

        // Compute residual vector
        TICK();
        double time1 = x.time1_, time2 = x.time2_;
        ierr = ComputeSPMV_ref(A, x, *A.mgData->Axf);
        if (ierr != 0)
            return ierr;
        x.time1_ = time1;
        x.time2_ = time2;
        TOCK(x.time1_);

        // Restriction operation
        TICK();
        ierr = ComputeRestriction_ref(A, r);
        if (ierr != 0) {
            HPGMP_RANGE_POP(__FUNCTION__);
            return ierr;
        }
        TOCK(x.time3_);

        // MG on coarser-grid
        A.mgData->xc->time1_ = A.mgData->xc->time2_ = 0.0;
        A.mgData->xc->time3_ = A.mgData->xc->time4_ = 0.0;
        ierr                                        = ComputeMG(*A.Ac, *A.mgData->rc, *A.mgData->xc, symmetric, ft);
        if (ierr != 0) {
            HPGMP_RANGE_POP(__FUNCTION__);
            return ierr;
        }
        x.time1_ += A.mgData->xc->time1_;
        x.time2_ += A.mgData->xc->time2_;
        x.time3_ += A.mgData->xc->time3_;
        x.time4_ += A.mgData->xc->time4_;

        // Prolongation operation
        TICK();
        ierr = ComputeProlongation_ref(A, x);
        if (ierr != 0) {
            HPGMP_RANGE_POP(__FUNCTION__);
            return ierr;
        }
        TOCK(x.time4_);

        // Post-smoothing
        const int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;
        if (symmetric) {
            for (int i = 0; i < numberOfPostsmootherSteps; ++i)
                ierr += ComputeSYMGS_ref(A, r, x);
        } else {
            for (int i = 0; i < numberOfPostsmootherSteps; ++i)
                ierr += ComputeGS_Forward_ref(A, r, x);
        }
        if (ierr != 0) {
            HPGMP_RANGE_POP(__FUNCTION__);
            return ierr;
        }
    } else {
        // coarsest grid
        if (symmetric) {
            ierr = ComputeSYMGS_ref(A, r, x);
        } else {
            ierr = ComputeGS_Forward_ref(A, r, x);
        }
        if (ierr != 0) {
            HPGMP_RANGE_POP(__FUNCTION__);
            return ierr;
        }
    }

    HPGMP_RANGE_POP(__FUNCTION__);

    return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

template int ComputeMG< SparseMatrix<double>, Vector<double> >(
    SparseMatrix<double> const&, Vector<double> const&, Vector<double>&, bool, perf_counters&);

template int ComputeMG< SparseMatrix<float>, Vector<float> >(
    SparseMatrix<float> const&, Vector<float> const&, Vector<float>&, bool, perf_counters&);
