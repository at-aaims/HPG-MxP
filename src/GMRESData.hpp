
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
 @file CGData.hpp

 HPGMP data structure
 */

#ifndef GMRESDATA_HPP
#define GMRESDATA_HPP

#include <array>
#include "Vector.hpp"

template <class SC, class PSC = SC>
class GMRESData {
public:
  using scalar_type  = SC;
  using project_type = PSC;
  Vector<SC> r; //!< pointer to residual vector
  Vector<SC> z; //!< pointer to preconditioned residual vector
  Vector<SC> p; //!< pointer to direction vector
  Vector<SC> w; //!< pointer to workspace
  Vector<SC> Ap; //!< pointer to Krylov vector
};

/*!
 Constructor for the data structure of GMRES vectors.

 @param[in]  A    the data structure that describes the problem matrix and its structure
 @param[out] data the data structure for GMRES  vectors that will be allocated to get it ready for use in GMRES iterations
 */
template <class SparseMatrix_type, class GMRESData_type>
inline void InitializeSparseGMRESData(SparseMatrix_type & A, GMRESData_type & data) {
  local_int_t nrow = A.localNumberOfRows;
  local_int_t ncol = A.localNumberOfColumns;
  comm_type comm = A.comm;

  InitializeVector(data.r,  nrow, comm);
  InitializeVector(data.z,  ncol, comm);
  InitializeVector(data.p,  ncol, comm);
  InitializeVector(data.w,  nrow, comm);
  InitializeVector(data.Ap, nrow, comm);
  return;
}

/*!
 Destructor for the GMRES vectors data.

 @param[inout] data the GMRES vectors data structure whose storage is deallocated
 */
template <class GMRESData_type>
inline void DeleteGMRESData(GMRESData_type & data) {

  DeleteVector (data.r);
  DeleteVector (data.z);
  DeleteVector (data.p);
  DeleteVector (data.w);
  DeleteVector (data.Ap);
  return;
}



template<class SC>
class TestGMRESData {
public:
  static constexpr int n_fl_ops = 4;     ///< Number of numerical operations for which flops are counted in GMRES
  static constexpr int n_timed_ops = 12; ///< Number of operations which are timed in GMRES

  int restart_length;     //!< restart length
  SC tolerance;           //!< tolerance = reference residual norm 
  double runningTime;     //!<
  double minOfficialTime; //!<

  // from validation step
  int refNumIters;       //!< number of reference iterations
  int optNumIters;       //!< number of optimized iterations
  int validation_nprocs; //!<
  double refResNorm0;
  double refResNorm;
  double optResNorm0;
  double optResNorm;

  // setup time
  double SetupTime;
  double OptimizeTime;
  double SpmvMgTime;

  // from benchmark step
  int numOfCalls;       //!< number of calls
  int maxNumIters;      //!< 
  int numOfMGCalls;     //!< number of MG calls
  int numOfSPCalls;     //!< number of SpMV calls
  int optNumOfMGCalls;  //!< number of MG calls   (opt)
  int optNumOfSPCalls;  //!< number of SpMV calls (opt)
  int refNumOfMGCalls;  //!< number of MG calls   (ref)
  int refNumOfSPCalls;  //!< number of SpMV calls (ref)
  double refTotalFlops; //
  double refTotalTime;  //
  double optTotalFlops; //
  double optTotalTime;  //
  //!< flop counts and time for total, dot, axpy, ortho, spmv, reduce, precond
  std::array<double,n_fl_ops> flops;        //!< accumulated in GMRES, temporary workspace
  std::array<double,n_timed_ops> times;        //!< accumulated in GMRES, temporary workspace
  std::array<double,n_timed_ops> times_comp;   //!< accumulated in GMRES, temporary workspace
  std::array<double,n_timed_ops> times_comm;   //!< accumulated in GMRES, temporary workspace

  std::array<double,n_fl_ops> ref_flops;    //!< record from output of reference GMRES
  std::array<double,n_timed_ops> ref_times;    //!< record from output of reference GMRES
  std::array<double,n_fl_ops> opt_flops;    //!< record from output of optimized GMRES
  std::array<double,n_timed_ops> opt_times;    //!< record from output of optimized GMRES
  std::array<double,n_timed_ops> ref_times_comp; //!< record from output of reference GMRES
  std::array<double,n_timed_ops> ref_times_comm; //!< record from output of reference GMRES
  std::array<double,n_timed_ops> opt_times_comp; //!< record from output of optimized GMRES
  std::array<double,n_timed_ops> opt_times_comm; //!< record from output of optimized GMRES
};

#endif // CGDATA_HPP

