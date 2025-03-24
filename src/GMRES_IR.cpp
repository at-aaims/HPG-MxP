
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
 @file GMRES_IR.cpp

 GMRES-IR routine
 */

#include <fstream>
#include <cmath>

#include "hpgmp.hpp"

#include "GMRES_IR.hpp"
#include "mytimer.hpp"
#include "ComputeSPMV.hpp"
#include "ComputeMG.hpp"
#include "ComputeDotProduct.hpp"
#include "ComputeWAXPBY_opt.hpp"
#include "ComputeTRSM.hpp"
#include "ComputeGEMV.hpp"
#include "ComputeGEMVT.hpp"
#include "ComputeGEMMT.hpp"


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

  @see GMRES_IR_ref()
*/
template<class SparseMatrix_type, class SparseMatrix_type2, class GMRESData_type, class GMRESData_type2, class Vector_type, class TestGMRESData_type>
int GMRES_IR(const SparseMatrix_type & A, const SparseMatrix_type2 & A_lo,
             GMRESData_type & data, GMRESData_type2 & data_lo, const Vector_type & b_hi, Vector_type & x_hi,
             const int restart_length, const int max_iter, const typename SparseMatrix_type::scalar_type tolerance,
             int & niters, typename SparseMatrix_type::scalar_type & normr_hi, typename SparseMatrix_type::scalar_type & normr0_hi,
             const bool doPreconditioning, bool verbose, TestGMRESData_type & test_data) {

  // (working) precision for outer loop
  typedef typename SparseMatrix_type::scalar_type scalar_type;
  typedef MultiVector<scalar_type> MultiVector_type;
  //typedef SerialDenseMatrix<scalar_type> SerialDenseMatrix_type;
  // (lower) precision for inner loop
  typedef typename SparseMatrix_type2::scalar_type scalar_type2;
  typedef MultiVector<scalar_type2> MultiVector_type2;
  //typedef SerialDenseMatrix<scalar_type2> SerialDenseMatrix_type;
  typedef Vector<scalar_type2> Vector_type2;
  // (lower) precision for storing projected matrix
  typedef typename GMRESData_type2::project_type project_type;
  typedef SerialDenseMatrix<project_type> SerialDenseMatrix_type;

  double start_t = 0.0, t0 = 0.0, t1 = 0.0, t1_ = 0.0, t2 = 0.0, t3 = 0.0, t3_1 = 0.0, t3_2 = 0.0,
	 t4 = 0.0, t5 = 0.0, t6 = 0.0, t7 = 0.0, t8 = 0.0, t9 = 0.0, t10 = 0.0, t11 = 0.0;
  double t1_comp = 0.0, t1_comm = 0.0;

  // vectors/matrices in scalar_type2 (lower)
  const global_int_t ione  = 1;
  const global_int_t itwo  = 2;
  const global_int_t ifour = 4;
  const scalar_type2 one  (1.0);
  const scalar_type2 zero (0.0);
  const project_type one_pr  (1.0);
  const project_type two_pr  (2.0);
  const project_type zero_pr (0.0);
  project_type normr, normr0;
  project_type alpha = zero_pr, beta = zero_pr;

  local_int_t  nrow = A_lo.localNumberOfRows;
  global_int_t Nrow = A.totalNumberOfRows;
  //Vector_type2 & x = data_lo.w; // Intermediate solution vector
  Vector_type2 & r = data_lo.r; // Residual vector
  Vector_type2 & z = data_lo.z; // Preconditioned residual vector
  //Vector_type2 & p = data_lo.p; // Direction vector (in MPI mode ncol>=nrow)
  //Vector_type2 & Ap = data_lo.Ap;

  SerialDenseMatrix_type H (restart_length+1, restart_length, x_hi.get_device_context());
  SerialDenseMatrix_type h (restart_length+1, 1, x_hi.get_device_context());
  SerialDenseMatrix_type t (restart_length+1, 1, x_hi.get_device_context());
  SerialDenseMatrix_type cs(restart_length+1, 1, x_hi.get_device_context());
  SerialDenseMatrix_type ss(restart_length+1, 1, x_hi.get_device_context());

  MultiVector_type2 Q(nrow, restart_length+1, A.comm, x_hi.get_device_context());
  //#define SINGLEREDUCE_GMRES_IR
  #ifdef SINGLEREDUCE_GMRES_IR
  MultiVector_type2 V;
  SerialDenseMatrix_type T( restart_length, restart_length);
  // workspaces
  SerialDenseMatrix_type G( restart_length+1, 2);
  SerialDenseMatrix_type w( restart_length+1, 1);
  #endif

  // vectors in scalar_type (higher)
  const scalar_type zero_hi (0.0);
  const scalar_type one_hi  (1.0);
  Vector_type & r_hi = data.r; // Residual vector
  Vector_type & z_hi = data.z; // Preconditioned residual vector
  Vector_type & p_hi = data.p; // Direction vector (in MPI mode ncol>=nrow)
  Vector_type & Ap_hi = data.Ap;

  if (!doPreconditioning && A.geom->rank==0) HPGMP_fout << "WARNING: PERFORMING UNPRECONDITIONED ITERATIONS" << std::endl;

  int print_freq = 1;
  if (print_freq>50) print_freq=50;
  if (print_freq<1)  print_freq=1;
  if (verbose && A.geom->rank==0) {
    HPGMP_fout << std::endl << " Running GMRES_IR(" << restart_length
                           << ") with max-iters = " << max_iter
                           << ", tol = " << tolerance
                           << " and restart = " << restart_length
                           << (doPreconditioning ? " with precond" : " without precond")
                           << ", nrow = " << nrow << std::endl;
    if (std::is_same<scalar_type2, double>::value) 
      HPGMP_fout << " Inner-Iter precision : double" << std::endl;
    if (std::is_same<scalar_type2, float>::value) 
      HPGMP_fout << " Inner-Iter precision : float" << std::endl;
    if (std::is_same<project_type, double>::value) 
      HPGMP_fout << " Projection precision : double" << std::endl;
    if (std::is_same<project_type, float>::value) 
      HPGMP_fout << " Projection precision : float" << std::endl;
  }
  double flops = 0.0;
  double flops_gmg  = 0.0;
  double flops_spmv = 0.0;
  double flops_orth = 0.0;
  global_int_t numSpMVs_MG = 1+(A.mgData->numberOfPresmootherSteps + A.mgData->numberOfPostsmootherSteps);
  niters = 0;
  bool converged = false;
  double t_begin = mytimer();  // Start timing right away
  while (niters <= max_iter && !converged) {
    // > Compute residual vector (higher working precision)
    // p is of length ncols, copy x to p for sparse MV operation
    CopyVector(x_hi, p_hi);
    TICK(); ComputeSPMV(A, p_hi, Ap_hi); flops_spmv += (2*A.totalNumberOfNonzeros); TOCK(t3); t3_1 += p_hi.time1; t3_2 += p_hi.time2; // Ap = A*p
    TICK(); ComputeWAXPBY_opt(nrow, one_hi, b_hi, -one_hi, Ap_hi, r_hi, A.isWaxpbyOptimized); flops += (itwo*Nrow);  TOCK(t11); // r = b - Ax (x stored in p)
    TICK(); ComputeDotProduct(nrow, r_hi, r_hi, normr_hi, t4, A.isDotProductOptimized); flops += (itwo*Nrow); TOCK(t11);
    normr_hi = sqrt(normr_hi);
    test_data.numOfSPCalls++;
    // Record initial residual for convergence testing
    if (niters == 0) {
      normr0 = normr_hi;
      normr0_hi = normr_hi;
    }
    normr = normr_hi;

    // Convergence check
    #define HPGMP_NUMERIC_CHECK
    #ifdef HPGMP_NUMERIC_CHECK
    project_type ortho_err (0.0);
    #endif
    if (verbose && A.geom->rank==0) {
      HPGMP_fout << "GMRES_IR Residual at the start of restart cycle = "
                << normr_hi << " / " << normr0_hi << " = " << normr_hi/normr0_hi
                << ", H(0,0) = " << normr << std::endl;
      HPGMP_fout << "GMRES_IR Iteration = "<< 0 << " (" << niters << ")   Scaled Computed Residual = "
                << normr_hi << " / " << normr0_hi << " = " << normr_hi/normr0_hi;
      #ifdef HPGMP_NUMERIC_CHECK
      HPGMP_fout << " (True Residual = " << normr_hi << " / " << normr0_hi << " = " << normr_hi/normr0_hi << ")";
      HPGMP_fout << "  Ortho Error = " << 0.0;
      #endif
      HPGMP_fout << std::endl;
    }
    if (IS_NAN(normr)) {
      break;
    }
    if (normr_hi/normr0_hi <= tolerance) { // Use "<=" to exit when res=zero (continuing will cause NaN)
      converged = true;
      if (verbose && A.geom->rank==0) HPGMP_fout << " > GMRES_IR converged " << std::endl;
      break;
    }

    // > Scale to the residual vector in working precision
    TICK();
    //ScaleVectorValue<Vector_type, scalar_type> (r_hi, one_hi/normr_hi);
    r_hi.scale(one_hi/normr_hi);
    flops += Nrow; TOCK(t11);

    // > Copy r as the initial basis vector (lower precision)
    auto Qj = Q.get_vector(0);
    CopyVector(r_hi, Qj);

    // do forward GS instead of symmetric GS
    const bool symmetric = false;

    // Start restart cycle
    global_int_t k = 1;
    t.set_value(0, 0, normr);
    while (k <= restart_length && normr/normr0 > tolerance && !IS_NAN(normr))
    {
      // Use ">" to exit when res=zero (continuing will cause NaN)
      auto Qkm1 = Q.get_vector(k-1);
      auto Qk = Q.get_vector(k);

      TICK();
      if (doPreconditioning) {
        z.time1 = z.time2 = z.time3 = z.time4 = 0.0;
        ComputeMG(A_lo, Qkm1, z, symmetric); flops_gmg += (2*numSpMVs_MG*A.totalNumberOfMGNonzeros); // Apply preconditioner
        test_data.numOfMGCalls++;
        t7 += z.time1; t8 += z.time2; t9 += z.time3; t10 += z.time4;
      } else {
        CopyVector(Qkm1, z);       // copy r to z (no preconditioning)
      }
      TOCK(t5); // Preconditioner apply time

      // Qk = A*z
      TICK(); ComputeSPMV(A_lo, z, Qk); flops_spmv += (2*A.totalNumberOfNonzeros); TOCK(t3); t3_1 += z.time1; t3_2 += z.time2;
      test_data.numOfSPCalls++;

      // orthogonalize z against Q(:,0:k-1), using dots
      bool use_mgs = false;
      TICK();
      if (use_mgs) {
        // MGS2
        for (int j = 0; j < k; j++) {
          // get j-th column of Q
          auto Qj = Q.get_vector(j);

          alpha = zero_pr;
          for (int i = 0; i < 2; i++) {
            // beta = Qk'*Qj
            START_T(); ComputeDotProduct<Vector_type2, project_type>
                         (nrow, Qk, Qj, beta, t4, A.isDotProductOptimized); STOP_T(t1);

            // Qk = Qk - beta * Qj
            START_T(); ComputeWAXPBY_opt(nrow, one, Qk, -beta, Qj, Qk, A.isWaxpbyOptimized); STOP_T(t2);
            alpha += beta;
          }
          H.set_value(j, k-1, alpha);
        }
        flops_orth += (ifour*k*Nrow);
      } else {
        // CGS2
        // first orthogonalization
        auto P = Q.get_multi_vector(0, k-1);
        // h = Q(1:k)'*q(k+1), mul and add in proj_type
        START_T(); ComputeGEMVT (nrow, k,  one, P, Qk, zero_pr, h, A.isGemvOptimized); STOP_T(t1);
        START_T(); ComputeGEMV  (nrow, k, -one, P, h,  one,    Qk, A.isGemvOptimized); STOP_T(t2); // q(k+1) = q(k+1) - Q(1:k)*h
        t1_comp += h.time1; t1_comm += h.time2;
        for(int i = 0; i < k; i++) {
          H.set_value(i, k-1, h.values()[i]);
        }
        flops_orth += (ifour*k*Nrow);

        START_T();
        // reorthogonalization
        // h = Q(1:k)'*q(k+1)
        ComputeGEMVT (nrow, k,  one, P, Qk, zero_pr, h, A.isGemvOptimized);
        STOP_T(t1);

        START_T(); ComputeGEMV (nrow, k, -one, P, h,  one, Qk, A.isGemvOptimized); STOP_T(t2); // q(k+1) = q(k+1) - Q(1:k)*h
        t1_comp += h.time1; t1_comm += h.time2;
        for(int i = 0; i < k; i++) {
          H.add_value(i, k-1, h.values()[i]);
        }
        flops_orth += (ifour*k*Nrow);
      } // end or CGS2

      // beta = norm(Qk)
      START_T(); ComputeDotProduct<Vector_type2, project_type>(nrow, Qk, Qk, beta, t4, A.isDotProductOptimized); STOP_T(t1_);
      flops_orth += (itwo*Nrow);
      beta = sqrt(beta);

      // Qk = Qk / beta
      // NOTE: Qk is scalar_type2, so the scaling factor is cast to this type before scaling.
      START_T(); Qk.scale(static_cast<scalar_type2>(one_pr/beta)); STOP_T(t2);
      flops_orth += (Nrow);

      TOCK(t6); // Ortho time
      H.set_value(k, k-1, beta);
      #if 0
      for (int i = 0; i <= k; ++i) HPGMP_fout << " + h[" << i << "] = " << GetMatrixValue(H, i, k-1) << std::endl;
      HPGMP_fout << std::endl;
      #endif

      // Given's rotation
      for(int j = 0; j < k-1; j++){
        const auto cj = static_cast<project_type>(cs.get_value(j, 0));
        const auto sj = static_cast<project_type>(ss.get_value(j, 0));
        const auto h1 = static_cast<project_type>(H.get_value(j,   k-1));
        const auto h2 = static_cast<project_type>(H.get_value(j+1, k-1));

        H.set_value(j+1, k-1, -sj * h1 + cj * h2);
        H.set_value(j,   k-1,  cj * h1 + sj * h2);
      }

      const auto f = static_cast<project_type>(H.get_value(k-1, k-1));
      const auto g = static_cast<project_type>(H.get_value(k,   k-1));

      const project_type f2 = f*f;
      const project_type g2 = g*g;
      project_type fg2 = f2 + g2;
      const project_type D1 = one_pr / sqrt(f2*fg2);
      const project_type cj = f2*D1;
      fg2 = fg2 * D1;
      const project_type sj = f*D1*g;
      H.set_value(k-1, k-1, f*fg2);
      H.set_value(k,   k-1, zero_pr);

      const auto v1 = static_cast<project_type>(t.get_value(k-1, 0));
      const project_type v2 = -v1*sj;
      t.set_value(k,   0, v2);
      t.set_value(k-1, 0, v1*cj);

      ss.set_value(k-1, 0, sj);
      cs.set_value(k-1, 0, cj);

      normr = std::abs(v2);
      if (verbose && (k%print_freq == 0 || k+1 == restart_length)) {
        #ifdef HPGMP_NUMERIC_CHECK
        {
          // compute current approximation
          CopyVector(x_hi, p_hi);                                 // using p_hi for x_hi
          for (int i=0; i <= k; i++) {
              h.set_value(i,0, t.values()[i]);   // using h for t
          }
          ComputeTRSM(k, one_pr, H, h);
          if (doPreconditioning) {
            #ifdef HPGMRES_IR_UPDATE_X_IN_HIGH
            ComputeGEMV(nrow, k, one, Q, h, zero_hi, r_hi, A.isGemvOptimized);          // r = Q*t (using h for t)
            ComputeMG(A_lo, r_hi, z_hi, symmetric);                                     // z = M*r
            ComputeWAXPBY_opt(nrow, one_hi, p_hi, one_hi, z_hi, p_hi, A.isWaxpbyOptimized); // x += z
            #else
            ComputeGEMV(nrow, k, one, Q, h, zero, r, A.isGemvOptimized);             // r = Q*t (using h for t)
            ComputeMG(A_lo, r, z, symmetric);                                        // z = M*r
            ComputeWAXPBY_opt(nrow, one_hi, p_hi, one, z, p_hi, A.isWaxpbyOptimized);    // x += z
            #endif
          } else {
            ComputeGEMV (nrow, k, one_hi, Q, h, one_hi, p_hi, A.isGemvOptimized);    // x += Q*t
          }
          // compute residual norm
          ComputeSPMV(A, p_hi, Ap_hi); // Ap = A*p
          ComputeWAXPBY_opt(nrow, one_hi, b_hi, -one_hi, Ap_hi, r_hi, A.isWaxpbyOptimized); // r = b - Ax (x stored in p)
          ComputeDotProduct(nrow, r_hi, r_hi, normr_hi, t4, A.isDotProductOptimized);
          normr_hi = sqrt(normr_hi);
        }
        {
          auto P = Q.get_multi_vector(0, k);
          for (int j=0; j<=k; j++) {
            auto Qk = Q.get_vector(j);
            ComputeGEMVT (nrow, k+1, one, P, Qk, zero_pr, h, A.isGemvOptimized);
            for (int i=0; i<=k; i++) {
              project_type error_i = (i == j ? h.values()[i]-one_pr : h.values()[i]);
              error_i = std::abs(error_i);
              ortho_err = (error_i > ortho_err ? error_i : ortho_err);
              //if (std::is_same<scalar_type, double>::value && std::is_same<project_type, float>::value && doPreconditioning) {
              //if (verbose && A.geom->rank==0 && k == restart_length) HPGMP_fout << " " << (i == j ? h.values()[i]-one_pr : h.values()[i]);
              //}
            } 
          }
        }
        #endif // HPGMP_NUMERIC_CHECK
        if (verbose && A.geom->rank==0) {
          HPGMP_fout << "GMRES_IR Iteration = "<< k << " (" << niters << ")   Scaled Computed Residual = "
                    << normr << " / " << normr0 << " = " << normr/normr0;
          #ifdef HPGMP_NUMERIC_CHECK
          HPGMP_fout << " (True Residual = " << normr_hi << " / " << normr0_hi << " = " << normr_hi/normr0_hi << ")";
          HPGMP_fout << "  Ortho Error = " << ortho_err;
          #endif
          HPGMP_fout << std::endl;
        }
      }
      niters ++;
      k ++;
    } // end of restart-cycle

    // prepare to restart
    if (verbose && A.geom->rank==0)
      HPGMP_fout << "GMRES_IR restart: k = "<< k << " (" << niters << ")" << std::endl;
    // > update x
    ComputeTRSM(k-1, one_pr, H, t);
    if (doPreconditioning) {
      #ifdef HPGMRES_IR_UPDATE_X_IN_HIGH
      ComputeGEMV (nrow, k-1, one, Q, t, zero_hi, r_hi, A.isGemvOptimized); flops += (itwo*Nrow*(k-ione)); // r = Q*t

      z.time1 = z.time2 = z.time3 = z.time4 = 0.0;
      TICK();
      ComputeMG(A, r_hi, z_hi, symmetric); flops_gmg += (2*numSpMVs_MG*A.totalNumberOfMGNonzeros);    // z = M*r
      TOCK(t5); // Preconditioner apply time
      test_data.numOfMGCalls++;
      t7 += z.time1; t8 += z.time2; t9 += z.time3; t10 += z.time4;

      // (mixed-precision) x += z
      TICK(); ComputeWAXPBY_opt(nrow, one_hi, x_hi, one_hi, z_hi, x_hi, A.isWaxpbyOptimized); flops += (itwo*Nrow); TOCK(t11);
      #else
      ComputeGEMV (nrow, k-1, one, Q, t, zero, r, A.isGemvOptimized); flops += (itwo*Nrow*(k-ione)); // r = Q*t

      z.time1 = z.time2 = z.time3 = z.time4 = 0.0;
      TICK();
      ComputeMG(A_lo, r, z, symmetric); flops_gmg += (2*numSpMVs_MG*A.totalNumberOfMGNonzeros);    // z = M*r
      TOCK(t5); // Preconditioner apply time
      test_data.numOfMGCalls++;
      t7 += z.time1; t8 += z.time2; t9 += z.time3; t10 += z.time4;

      // mixed-precision
      TICK(); ComputeWAXPBY_opt(nrow, one_hi, x_hi, one, z, x_hi, A.isWaxpbyOptimized); flops += (itwo*Nrow); TOCK(t11); // x += z
      #endif
    } else {
      // mixed-precision
      ComputeGEMV (nrow, k-1, one_hi, Q, t, one_hi, x_hi, A.isGemvOptimized); flops += (itwo*Nrow*(k-ione)); // x += Q*t
    }
  } // end of outer-loop


  // Store times
  double tt = mytimer() - t_begin;
  test_data.times[0]  += tt;       // Total time. All done...
  test_data.times[1]  += t1 + t1_; // dot-product time
  test_data.times[2]  += t2;       // WAXPBY time
  test_data.times[3]  += t6;       // Ortho
  test_data.times[4]  += t3;       // SPMV time
  test_data.times[5]  += t4;       // AllReduce time
  test_data.times[6]  += t5;       // preconditioner apply time
  test_data.times[7]  += t7;       // > SpTRSV for GS
  test_data.times[8]  += t8;       // > SpMV for GS
  test_data.times[9]  += t9;       // > Restrict for GS
  test_data.times[10] += t10;      // > Prolong for GS
  test_data.times[11] += t11;      // Vector update time

  test_data.times_comp[1] += t1_comp; // dot-product time
  test_data.times_comm[1] += t1_comm; // dot-product time
  test_data.times_comm[2] += t3_1;     // > SPMV local copy
  test_data.times_comm[3] += t3_2;     // > SPMV halo exchange
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

  //return ((converged && !IS_NAN(normr)) ? 0 : 1);
  if(!converged) {
      HPGMP_fout << "GMRES-IR did not converge!\n";
      return 1;
  } else if (IS_NAN(normr)) {
      HPGMP_fout << "GMRES-IR resulted in NaN!\n";
      return 2;
  } else {
      return 0;
  }
}


/* --------------- *
 * specializations *
 * --------------- */

// uniform
template
int GMRES_IR< SparseMatrix<double>, SparseMatrix<double>, GMRESData<double>, GMRESData<double>, Vector<double>, TestGMRESData<double> >
  (SparseMatrix<double> const&, SparseMatrix<double> const&, GMRESData<double>&, GMRESData<double>&,
   Vector<double> const&, Vector<double>&, const int, const int, double, int&, double&, double&, bool, bool,
   TestGMRESData<double>&);

template
int GMRES_IR< SparseMatrix<float>, SparseMatrix<float>, GMRESData<float>, GMRESData<float>, Vector<float>, TestGMRESData<float> >
  (SparseMatrix<float> const&, SparseMatrix<float> const&, GMRESData<float>&, GMRESData<float>&,
   Vector<float> const&, Vector<float>&, const int, const int, float, int&, float&, float&, bool, bool,
   TestGMRESData<float>&);


// mixed
template
int GMRES_IR< SparseMatrix<double>, SparseMatrix<float>, GMRESData<double>, GMRESData<float>, Vector<double>, TestGMRESData<double> >
  (SparseMatrix<double> const&, SparseMatrix<float> const&, GMRESData<double>&, GMRESData<float>&,
   Vector<double> const&, Vector<double>&, const int, const int, double, int&, double&, double&, bool, bool,
   TestGMRESData<double>&);
