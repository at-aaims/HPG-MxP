
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
 @file ComputeMG.cpp

 HPGMP routine
 */

#include "ComputeMG.hpp"

#include <cassert>

#include "ell_multicolor_gs.hpp"
#include "restriction.hpp"
#include "prolongation.hpp"
#include "mytimer.hpp"

/*!
 * @param[in] A the known system matrix
 * @param[in] r the residual or input RHS on the finest grid.
 * @param[inout] x On exit contains the result of the multigrid V-cycle
 *                 x is the approximation to Ax = r.
 *                 Also contains timing information of the whole multigrid preconditioner on exit.
 * @param[inout] ft  It is updated with the flops, memory traffic etc. for the global multigrid
 *                   preconditioner.
 *
 * The total number of flops in the entire MG preconditioner with n levels is
 * 2*nnz_allgrids*(1 + n_presmooth + n_postsmooth) - 2*(nnz_finest - nnz_coarsest).
 * Note that this is cheaper than the reference ComputeMG_ref because of fusing the residual
 * calculation and restriction.
 *
 * @return returns 0 upon success and non-zero otherwise
 *
 * @see ComputeMG_ref
 */
template<class SparseMatrix_type, class Vector_type>
int ComputeMG(const SparseMatrix_type & A, const Vector_type & r, Vector_type & x,
              const bool symmetric, flops_and_traffic& ft)
{
  using scalar_type = typename SparseMatrix_type::scalar_type;
  const int mpisize = A.geom->size;

  // Optimized versions of calls
  double t0 = 0.0;
  x.fill_zero();
  ft.f_mem_traffic[0] += x.local_length();

  std::shared_ptr<const ELLMatrix<scalar_type>> mat =
      dynamic_cast<EllOptData<scalar_type>*>(A.optimizationData)->mat;

  int ierr = 0;
  if (A.mgData!=0)
  {
    const int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;

    // Intentional abuse of Vector class timing variables
    auto time_gs_sofar = x.time2_;
    x.time2_ = 0.0;
    TICK();

    for(int i=0; i < numberOfPresmootherSteps; ++i) {
      if(i = 0) {
        ierr += ell_multicolor_gs_zero_initial(symmetric, mat.get(), &r, &x);
        ft.flops[0] += A.totalNumberOfNonzeros;
      } else {
        ierr += ell_multicolor_gs(symmetric, mat.get(), &r, &x);
        ft.flops[0] += 2*A.totalNumberOfNonzeros;
      }
    }
    x.time1_ = 0.0;  // Apparently no one cares about GS comm time
    // restore GS time so far
    x.time2_ = time_gs_sofar;
    TOCK(x.time2_);

    if (ierr!=0)
        return ierr;

    // Restriction operation
    TICK();
    //ierr = restriction(A, r);
    ierr = fused_spmv_restriction(A, r, x);
    ft.flops[0] += 2*A.Ac->totalNumberOfNonzeros;
    if (ierr!=0)
      return ierr;
    TOCK(x.time3_);

    // MG on coarser-grid
    A.mgData->xc->time1_ = A.mgData->xc->time2_ = 0.0;
    A.mgData->xc->time3_ = A.mgData->xc->time4_ = 0.0;
    ierr = ComputeMG(*A.Ac,*A.mgData->rc, *A.mgData->xc, symmetric, ft);
    if (ierr!=0)
      return ierr;
    x.time1_ += A.mgData->xc->time1_; x.time2_ += A.mgData->xc->time2_;
    x.time3_ += A.mgData->xc->time3_; x.time4_ += A.mgData->xc->time4_;

    // Prolongation operation
    TICK();
    ierr = prolongation(A, x);
    ft.flops[0] += mpisize*A.mgData->rc->local_length();
    if (ierr!=0)
      return ierr;
    TOCK(x.time4_);

    // Post-smoothing
    time_gs_sofar = x.time2_;
    x.time2_ = 0.0;
    TICK();

    const int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;
    for (int i=0; i< numberOfPostsmootherSteps; ++i) {
      ierr += ell_multicolor_gs(symmetric, mat.get(), &r, &x);
      ft.flops[0] += 2*A.totalNumberOfNonzeros;
    }
    x.time1_ = 0.0;  // Apparently no one cares about GS comm time
    // restore GS time so far
    x.time2_ = time_gs_sofar;
    TOCK(x.time2_);
    if (ierr!=0)
      return ierr;
  }
  else {
    // coarsest grid
    auto time_gs_sofar = x.time2_;
    x.time2_ = 0.0;
    TICK();

    ierr += ell_multicolor_gs(symmetric, mat.get(), &r, &x);

    x.time1_ = 0.0;  // Apparently no one cares about GS comm time
    // restore GS time so far
    x.time2_ = time_gs_sofar;
    TOCK(x.time2_);
    ft.flops[0] += 2*A.totalNumberOfNonzeros;
    if (ierr!=0)
      return ierr;
  }
  return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

template
int ComputeMG(SparseMatrix<double> const&, Vector<double> const&, Vector<double>&, bool,
              flops_and_traffic&);

template
int ComputeMG(SparseMatrix<float> const&, Vector<float> const&, Vector<float>&, bool,
              flops_and_traffic&);

