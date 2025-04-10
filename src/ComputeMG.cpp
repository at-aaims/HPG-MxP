
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
 * @param[in] r the input vector
 * @param[inout] x On exit contains the result of the multigrid V-cycle with r as the RHS,
 *                 x is the approximation to Ax = r.
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
    // Go to next coarse level if defined
    const int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;
    for(int i=0; i < numberOfPresmootherSteps; ++i) {
      if(i = 0) {
        ierr += ell_multicolor_gs_zero_initial(symmetric, mat.get(), &r, &x);
        ft.flops[0] += A.totalNumberOfNonzeros;
      } else {
        ierr += ell_multicolor_gs(symmetric, mat.get(), &r, &x);
        ft.flops[0] += 2*A.totalNumberOfNonzeros;
      }
    }
    if (ierr!=0)
        return ierr;

    // Restriction operation
    TICK();
    //ierr = restriction(A, r);
    ierr = fused_spmv_restriction(A, r, x);
    ft.flops[0] += 2*A.Ac->totalNumberOfNonzeros;
    if (ierr!=0)
      return ierr;
    TOCK(x.time3);

    // MG on coarser-grid
    A.mgData->xc->time1 = A.mgData->xc->time2 = 0.0;
    A.mgData->xc->time3 = A.mgData->xc->time4 = 0.0;
    ierr = ComputeMG(*A.Ac,*A.mgData->rc, *A.mgData->xc, symmetric, ft);
    if (ierr!=0)
      return ierr;
    x.time1 += A.mgData->xc->time1; x.time2 += A.mgData->xc->time2;
    x.time3 += A.mgData->xc->time3; x.time4 += A.mgData->xc->time4;

    // Prolongation operation
    TICK();
    ierr = prolongation(A, x);
    ft.flops[0] += mpisize*A.mgData->rc->local_length();
    if (ierr!=0)
      return ierr;
    TOCK(x.time4);

    // Post-smoothing
    const int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;
    for (int i=0; i< numberOfPostsmootherSteps; ++i) {
      ierr += ell_multicolor_gs(symmetric, mat.get(), &r, &x);
      ft.flops[0] += 2*A.totalNumberOfNonzeros;
    }
    if (ierr!=0)
      return ierr;
  }
  else {
    // coarsest grid
    ierr += ell_multicolor_gs(symmetric, mat.get(), &r, &x);
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

