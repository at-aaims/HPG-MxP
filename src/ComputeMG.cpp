
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

//#include "ComputeMG_ref.hpp"
#include "ComputeMG.hpp"

#include <cassert>

#include "ComputeSYMGS.hpp"
//#include "ComputeGS_Forward.hpp"
#include "ell_multicolor_gs.hpp"
#include "ComputeSPMV.hpp"
#include "ComputeRestriction_ref.hpp"
#include "ComputeProlongation_ref.hpp"
#include "mytimer.hpp"

/*!
  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On exit contains the result of the multigrid V-cycle with r as the RHS, x is the approximation to Ax = r.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeMG_ref
*/
template<class SparseMatrix_type, class Vector_type>
int ComputeMG(const SparseMatrix_type & A, const Vector_type & r, Vector_type & x, bool symmetric) {

  // This line and the next two lines should be removed and your version of ComputeSYMGS should be used.
  //A.isMgOptimized = false;
  //return ComputeMG_ref(A, r, x, symmetric);
  using scalar_type = typename SparseMatrix_type::scalar_type;

  // Optimized versions of calls
  // initialize x to zero
  double t0 = 0.0;
  x.fill_zero();

  std::shared_ptr<const ELLMatrix<scalar_type>> mat =
      dynamic_cast<EllOptData<scalar_type>*>(A.optimizationData)->mat;

  int ierr = 0;
  if (A.mgData!=0) { // Go to next coarse level if defined
    const int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;
    if (symmetric) {
      for (int i=0; i< numberOfPresmootherSteps; ++i)
          ierr += ComputeSYMGS(A, r, x);
    } else {
      for (int i=0; i< numberOfPresmootherSteps; ++i)
          //ierr += ComputeGS_Forward(A, r, x);
          if(i = 0) {
              ierr += ell_multicolor_gs_zero_initial(mat.get(), &r, &x);
          } else {
              ierr += ell_multicolor_gs(mat.get(), &r, &x);
          }
    }
    if (ierr!=0)
        return ierr;

    // Compute residual vector
    TICK();
    double time1 = x.time1, time2 = x.time2;
    ierr = ComputeSPMV(A, x, *A.mgData->Axf);
    if (ierr!=0)
        return ierr;
    x.time1 = time1; x.time2 = time2;
    TOCK(x.time1);

    // Restriction operation
    TICK();
    ierr = ComputeRestriction_ref(A, r);
    if (ierr!=0)
        return ierr;
    TOCK(x.time3);

    // MG on coarser-grid
    A.mgData->xc->time1 = A.mgData->xc->time2 = 0.0;
    A.mgData->xc->time3 = A.mgData->xc->time4 = 0.0;
    ierr = ComputeMG(*A.Ac,*A.mgData->rc, *A.mgData->xc, symmetric);
    if (ierr!=0)
        return ierr;
    x.time1 += A.mgData->xc->time1; x.time2 += A.mgData->xc->time2;
    x.time3 += A.mgData->xc->time3; x.time4 += A.mgData->xc->time4;

    // Prolongation operation
    TICK();
    ierr = ComputeProlongation_ref(A, x);
    if (ierr!=0)
        return ierr;
    TOCK(x.time4);

    // Post-smoothing
    const int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;
    if (symmetric) {
      for (int i=0; i< numberOfPostsmootherSteps; ++i)
          ierr += ComputeSYMGS(A, r, x);
    } else {
      for (int i=0; i< numberOfPostsmootherSteps; ++i) {
          //ierr += ComputeGS_Forward(A, r, x);
          ierr += ell_multicolor_gs(mat.get(), &r, &x);
      }
    }
    if (ierr!=0)
        return ierr;
  }
  else {
    // coarsest grid
    if (symmetric) {
      ierr = ComputeSYMGS(A, r, x);
    } else {
      //ierr = ComputeGS_Forward(A, r, x);
      ierr += ell_multicolor_gs(mat.get(), &r, &x);
    }
    if (ierr!=0) return ierr;
  }
  return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

template
int ComputeMG< SparseMatrix<double>, Vector<double> >(SparseMatrix<double> const&, Vector<double> const&, Vector<double>&, bool);

template
int ComputeMG< SparseMatrix<float>, Vector<float> >(SparseMatrix<float> const&, Vector<float> const&, Vector<float>&, bool);

