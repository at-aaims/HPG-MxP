
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
 @file GMRES.cpp

 GMRES routine
 */

#include <fstream>
#include <iostream>
#include <cmath>

#include "hpgmp.hpp"

#include "GMRES.hpp"
#include "mytimer.hpp"
#include "ComputeSPMV.hpp"
#include "ComputeMG.hpp"
#include "ComputeDotProduct.hpp"
#include "ComputeWAXPBY.hpp"
#include "ComputeTRSM.hpp"
#include "ComputeGEMV.hpp"
#include "ComputeGEMVT.hpp"


/*!
  Routine to compute an approximate solution to Ax = b

  @param[in]    geom The description of the problem's geometry.
  @param[inout] A    The known system matrix
  @param[inout] data The data structure with all necessary CG vectors preallocated
  @param[in]    b    The known right hand side vector
  @param[inout] x    On entry: the initial guess; on exit: the new approximate solution
  @param[in]    max_iter  The maximum number of iterations to perform, even if tolerance is not met.
  @param[in]    tolerance The stopping criterion to assert convergence: if norm of residual is <= to tolerance.
  @param[out]   niters    The number of iterations actually performed.
  @param[out]   normr     The 2-norm of the residual vector after the last iteration.
  @param[out]   normr0    The 2-norm of the residual vector before the first iteration.
  @param[out]   times     The 7-element vector of the timing information accumulated during all of the iterations.
  @param[in]    doPreconditioning The flag to indicate whether the preconditioner should be invoked at each iteration.

  @return Returns zero on success and a non-zero value otherwise.

  @see GMRES_ref()
*/
template<class SparseMatrix_type, class GMRESData_type, class Vector_type, class TestGMRESData_type>
int GMRES(const SparseMatrix_type & A, GMRESData_type & data, const Vector_type & b, Vector_type & x,
          const int restart_length, const int max_iter, const typename SparseMatrix_type::scalar_type tolerance,
          int & niters, typename SparseMatrix_type::scalar_type & normr,  typename SparseMatrix_type::scalar_type & normr0,
          bool doPreconditioning, bool verbose, TestGMRESData_type & test_data) {
 
  typedef typename SparseMatrix_type::scalar_type scalar_type;
  typedef MultiVector<scalar_type> MultiVector_type;
  typedef SerialDenseMatrix<scalar_type> SerialDenseMatrix_type;

  const scalar_type one  (1.0);
  const scalar_type zero (0.0);
  const global_int_t ione  = 1;
  const global_int_t itwo  = 2;
  const global_int_t ifour = 4;
  double start_t = 0.0, t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0, t5 = 0.0, t6 = 0.0,
         t7 = 0.0, t8 = 0.0, t9 = 0.0, t10 = 0.0, t11 = 0.0;
  double t1_comp = 0.0, t1_comm = 0.0;

  local_int_t  nrow = A.localNumberOfRows;
  global_int_t Nrow = A.totalNumberOfRows;
  int print_freq = 1;
  if (verbose && A.geom->rank==0) {
    HPGMP_fout << std::endl << " Running GMRES(" << restart_length
                           << ") with max-iters = " << max_iter
                           << ", tol = " << tolerance
                           << " and restart = " << restart_length
                           << (doPreconditioning ? " with precond" : " without precond")
                           << ", nrow = " << nrow << " (local) " << Nrow << " (global)"
                           << " on ( " << A.geom->npx << " x " << A.geom->npy << " x " << A.geom->npz
                           << " ) MPI grid "
                           << std::endl;
    HPGMP_fout << std::flush;
  }
  normr = 0.0;
  scalar_type alpha = zero, beta = zero;

//#ifndef HPGMP_NO_MPI
//  double t6 = 0.0;
//#endif
  Vector_type & r = data.r; // Residual vector
  Vector_type & z = data.z; // Preconditioned residual vector
  Vector_type & p = data.p; // Direction vector (in MPI mode ncol>=nrow)
  Vector_type & Ap = data.Ap;

  MultiVector_type Q(nrow, restart_length+1, A.comm, x.get_device_context());
  SerialDenseMatrix_type H (restart_length+1, restart_length, x.get_device_context());
  SerialDenseMatrix_type h (restart_length+1, 1, x.get_device_context());
  SerialDenseMatrix_type t (restart_length+1, 1, x.get_device_context());
  SerialDenseMatrix_type cs(restart_length+1, 1, x.get_device_context());
  SerialDenseMatrix_type ss(restart_length+1, 1, x.get_device_context());

  if (!doPreconditioning && A.geom->rank==0)
      HPGMP_fout << "WARNING: PERFORMING UNPRECONDITIONED ITERATIONS" << std::endl;

  double flops = 0.0;
  double flops_gmg  = 0.0;
  double flops_spmv = 0.0;
  double flops_orth = 0.0;
  const global_int_t numSpMVs_MG = 1+(A.mgData->numberOfPresmootherSteps + A.mgData->numberOfPostsmootherSteps);
  niters = 0;
  bool converged = false;
  double t_begin = mytimer();  // Start timing right away
  while (niters <= max_iter && !converged) {
    // p is of length ncols, copy x to p for sparse MV operation
    // In HIP/Cuda builds, this copies only device buffers.
    CopyVector(x, p);
    TICK(); ComputeSPMV(A, p, Ap); TOCK(t3); flops_spmv += (2*A.totalNumberOfNonzeros); // Ap = A*p
    TICK(); ComputeWAXPBY(nrow, one, b, -one, Ap, r, A.isWaxpbyOptimized); TOCK(t11); flops += (itwo*Nrow); // r = b - Ax (x stored in p)
    TICK(); ComputeDotProduct(nrow, r, r, normr, t4, A.isDotProductOptimized); flops += (itwo*Nrow); TOCK(t11);
    normr = sqrt(normr);
    auto Qj = Q.get_vector(0);
    CopyVector(r, Qj);
    TICK(); Qj.scale(one/normr); TOCK(t11); flops += Nrow;

    // Record initial residual for convergence testing
    if (niters == 0) normr0 = normr;
    if (verbose && A.geom->rank==0) {
      HPGMP_fout << "GMRES Residual at the start of restart cycle = "<< normr << " / " << normr0
                << " = " << normr/normr0 << std::endl;
    }
    if (IS_NAN(normr)) {
      break;
    }
    if (normr/normr0 <= tolerance) { // Use "<=" to exit when res=zero (continuing will cause NaN)
      converged = true;
      if (verbose && A.geom->rank==0) HPGMP_fout << " > GMRES converged " << std::endl;
    }

    // do forward GS instead of symmetric GS
    const bool symmetric = false;

    // Start restart cycle
    int k = 1;
    t.set_value(0, 0, normr);
    //HPGMP_VERBOSE_PRINT("GMRES: Starting restart cycle..");
    while (k <= restart_length && normr/normr0 > tolerance) { // Use ">" to exit when res=zero (continuing will cause NaN)
      auto Qkm1 = Q.get_vector(k-1);
      auto Qk = Q.get_vector(k);

      TICK();
      if (doPreconditioning) {
        z.time1 = z.time2 = z.time3 = z.time4 = 0.0;
        ComputeMG(A, Qkm1, z, symmetric);
        flops_gmg += (2*numSpMVs_MG*A.totalNumberOfMGNonzeros); // Apply preconditioner
        t7 += z.time1; t8 += z.time2; t9 += z.time3; t10 += z.time4;
      } else {
        CopyVector(Qkm1, z);              // copy r to z (no preconditioning)
      }
      TOCK(t5); // Preconditioner apply time

      // Qk = A*z
      TICK(); ComputeSPMV(A, z, Qk); flops_spmv += (2*A.totalNumberOfNonzeros); TOCK(t3);


      // orthogonalize z against Q(:,0:k-1), using dots
      TICK();
#if 0
        // MGS2
        for (int j = 0; j < k; j++) {
          // get j-th column of Q
          GetVector(Q, j, Qj);

          alpha = zero;
          for (int i = 0; i < 2; i++) {
            // beta = Qk'*Qj
            START_T(); ComputeDotProduct(nrow, Qk, Qj, beta, t4, A.isDotProductOptimized); STOP_T(t1);

            // Qk = Qk - beta * Qj
            START_T(); ComputeWAXPBY(nrow, one, Qk, -beta, Qj, Qk, A.isWaxpbyOptimized); STOP_T(t2);
            alpha += beta;
          }
          SetMatrixValue(H, j, k-1, alpha);
        }
        flops_orth += (ifour*k*Nrow);
#endif
        // CGS2
        auto P = Q.get_multi_vector(0, k-1);
        // Computes GEMV^T and copies output h to host
        START_T(); ComputeGEMVT (nrow, k,  one, P, Qk, zero, h, A.isGemvOptimized); STOP_T(t1); // h = Q(1:k)'*q(k+1)
        // Copies input h to device and Computes GEMV
        START_T(); ComputeGEMV  (nrow, k, -one, P, h,  one, Qk, A.isGemvOptimized); STOP_T(t2); // h = Q(1:k)'*q(k+1)
        t1_comp += h.time1; t1_comm += h.time2;
        for(int i = 0; i < k; i++) {
          H.set_value(i, k-1, h.values()[i]);
        }
        flops_orth += (ifour*k*Nrow);
        // reorthogonalize
        START_T(); ComputeGEMVT (nrow, k,  one, P, Qk, zero, h, A.isGemvOptimized); STOP_T(t1); // h = Q(1:k)'*q(k+1)
        START_T(); ComputeGEMV  (nrow, k, -one, P, h,  one, Qk, A.isGemvOptimized); STOP_T(t2); // h = Q(1:k)'*q(k+1)
        t1_comp += h.time1; t1_comm += h.time2;
        for(int i = 0; i < k; i++) {
          H.add_value(i, k-1, h.values()[i]);
        }
        flops_orth += (ifour*k*Nrow);
        // end CGS2

      // beta = norm(Qk)
      START_T(); ComputeDotProduct(nrow, Qk, Qk, beta, t4, A.isDotProductOptimized); STOP_T(t1);
      flops_orth += (itwo*Nrow);
      beta = sqrt(beta);

      // Qk = Qk / beta
      START_T(); Qk.scale(one/beta); STOP_T(t2);
      flops_orth += (Nrow);
      TOCK(t6); // Ortho time

      H.set_value(k, k-1, beta);

      // Given's rotation
      for(int j = 0; j < k-1; j++){
        const double cj = cs.get_value(j, 0);
        const double sj = ss.get_value(j, 0);
        const double h1 = H.get_value(j,   k-1);
        const double h2 = H.get_value(j+1, k-1);

        H.set_value(j+1, k-1, -sj * h1 + cj * h2);
        H.set_value(j,   k-1,  cj * h1 + sj * h2);
      }

      const double f = H.get_value(k-1, k-1);
      const double g = H.get_value(k,   k-1);

      const double f2 = f*f;
      const double g2 = g*g;
      double fg2 = f2 + g2;
      const double D1 = one / sqrt(f2*fg2);
      const double cj = f2*D1;
      fg2 = fg2 * D1;
      const double sj = f*D1*g;
      H.set_value(k-1, k-1, f*fg2);
      H.set_value(k,   k-1, zero);

      const double v1 = t.get_value(k-1, 0);
      const double v2 = -v1*sj;
      t.set_value(k,   0, v2);
      t.set_value(k-1, 0, v1*cj);

      ss.set_value(k-1, 0, sj);
      cs.set_value(k-1, 0, cj);

      normr = std::abs(v2);
      if (verbose && A.geom->rank==0 && (k%print_freq == 0 || k+1 == restart_length)) {
        HPGMP_fout << "GMRES Iteration = "<< k << " (" << niters << ")   Scaled Residual = "
                  << normr << " / " << normr0 << " = " << normr/normr0 << std::endl;
        //HPGMP_fout << "Flop count : GMG = " << flops_gmg << " SpMV = " << flops_spmv << " Ortho = " << flops_orth << std::endl;
      }
      niters ++;
      k ++;
    } // end of restart-cycle
    // prepare to restart
    if (verbose && A.geom->rank==0) {
      HPGMP_fout << "GMRES restart: k = "<< k << " (" << niters << ")" << std::endl;
    }
    // > update x
    ComputeTRSM(k-1, one, H, t);
    if (doPreconditioning) {
      // t is on host, so ComputeGEMV first copies it to device before computation
      ComputeGEMV(nrow, k-1, one, Q, t, zero, r, A.isGemvOptimized); flops += (itwo*Nrow*(k-ione)); // r = Q*t

      z.time1 = z.time2 = z.time3 = z.time4 = 0.0;
      TICK();
      ComputeMG(A, r, z, symmetric); flops_gmg += (2*numSpMVs_MG*A.totalNumberOfMGNonzeros);      // z = M*r
      TOCK(t5); // Preconditioner apply time
      t7 += z.time1; t8 += z.time2; t9 += z.time3; t10 += z.time4;

      TICK(); ComputeWAXPBY(nrow, one, x, one, z, x, A.isWaxpbyOptimized); TOCK(t11); flops += (itwo*Nrow); // x += z
    } else {
      ComputeGEMV(nrow, k-1, one, Q, t, one, x, A.isGemvOptimized); flops += (itwo*Nrow*(k-ione)); // x += Q*t
    }
  } // end of outer-loop


  // Store times
  double tt = mytimer() - t_begin;
  test_data.times[0]  += tt;  // Total time. All done...
  test_data.times[1]  += t1;  // dot-product time
  test_data.times[2]  += t2;  // WAXPBY time
  test_data.times[3]  += t6;  // Ortho
  test_data.times[4]  += t3;  // SPMV time
  test_data.times[5]  += t4;  // AllReduce time
  test_data.times[6]  += t5;  // preconditioner apply time
  test_data.times[7]  += t7;  // > SpTRSV for GS
  test_data.times[8]  += t8;  // > SpMV for GS
  test_data.times[9]  += t9;  // > Restrict for GS
  test_data.times[10] += t10; // > Prolong for GS
  test_data.times[11] += t11; // Vector update time

  test_data.times_comp[1] += t1_comp; // dot-product time
  test_data.times_comm[1] += t1_comm; // dot-product time
//#ifndef HPGMP_NO_MPI
//  times[6] += t6; // exchange halo time
//#endif
  double flops_tot = flops + flops_gmg + flops_spmv + flops_orth;
  if (verbose && A.geom->rank==0) {
    HPGMP_fout << " > nnz(A)  : " << A.totalNumberOfNonzeros << std::endl;
    HPGMP_fout << " > nnz(MG) : " << A.totalNumberOfMGNonzeros << " (" << numSpMVs_MG << ")" << std::endl;
    HPGMP_fout << " > SpMV : " << (flops_spmv / 1000000000.0) << " / " << t3 << " = "
                              << (flops_spmv / 1000000000.0) / t3 << " Gflop/s" << std::endl;
    HPGMP_fout << " > GMG  : " << (flops_gmg  / 1000000000.0) << " / " << t5 << " = "
                              << (flops_gmg  / 1000000000.0) / t5 << " Gflop/s" << std::endl;
    HPGMP_fout << " > Orth : " << (flops_orth / 1000000000.0) << " / " << t6 << " = "
                              << (flops_orth / 1000000000.0) / t6 << " Gflop/s" << std::endl;
    HPGMP_fout << " > Total: " << (flops_tot  / 1000000000.0) << " / " << tt << " = "
                              << (flops_tot  / 1000000000.0) / tt << " Gflop/s" << std::endl;
    HPGMP_fout << std::endl;
  }
  test_data.flops[0] += flops_tot;
  test_data.flops[1] += flops_gmg;
  test_data.flops[2] += flops_spmv;
  test_data.flops[3] += flops_orth;

  if(IS_NAN(normr)) {
      return 2;
  } else if (!converged) {
      return 1;
  } else
      return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

template
int GMRES< SparseMatrix<double>, GMRESData<double>, Vector<double>, TestGMRESData<double> >
  (SparseMatrix<double> const&, GMRESData<double>&, Vector<double> const&, Vector<double>&,
   const int, const int, double, int&, double&, double&, bool, bool, TestGMRESData<double>&);

template
int GMRES< SparseMatrix<float>, GMRESData<float>, Vector<float>, TestGMRESData<float> >
  (SparseMatrix<float> const&, GMRESData<float>&, Vector<float> const&, Vector<float>&,
   const int, const int, float, int&, float&, float&, bool, bool, TestGMRESData<float>&);

