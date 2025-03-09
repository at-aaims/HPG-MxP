
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
 @file hpgmp.hpp

 HPGMP data structures and functions
 */

#ifndef HPGMP_HPP
#define HPGMP_HPP

#include <fstream>
#include "Geometry.hpp"

extern std::ofstream HPGMP_fout;
extern std::ofstream HPGMP_vout;

struct HPGMP_Params_STRUCT {
  int comm_size; //!< Number of MPI processes in MPI_COMM_WORLD
  int comm_rank; //!< This process' MPI rank in the range [0 to comm_size - 1]
  int numThreads; //!< This process' number of threads
  local_int_t nx; //!< Number of processes in x-direction of 3D process grid
  local_int_t ny; //!< Number of processes in y-direction of 3D process grid
  local_int_t nz; //!< Number of processes in z-direction of 3D process grid
  int runningTime; //!< Number of seconds to run the timed portion of the benchmark
  int npx; //!< Number of x-direction grid points for each local subdomain
  int npy; //!< Number of y-direction grid points for each local subdomain
  int npz; //!< Number of z-direction grid points for each local subdomain
  int pz; //!< Partition in the z processor dimension, default is npz
  local_int_t zl; //!< nz for processors in the z dimension with value less than pz
  local_int_t zu; //!< nz for processors in the z dimension with value greater than pz
};
/*!
  HPGMP_Params is a shorthand for HPGMP_Params_STRUCT
 */
typedef HPGMP_Params_STRUCT HPGMP_Params;
#ifdef HPGMP_NO_MPI
  typedef int comm_type;
#else
  #include "mpi.h"
  typedef MPI_Comm comm_type;
#endif

extern int HPGMP_Init_Params(const char *title, int * argc_p, char ** *argv_p, HPGMP_Params & params, comm_type comm);
extern int HPGMP_Init_Params(int * argc_p, char ** *argv_p, HPGMP_Params & params, comm_type comm);
extern int HPGMP_Init(int * argc_p, char ** *argv_p);
extern int HPGMP_Finalize(void);

#define IS_NAN(a) (std::isinf(a) || std::isnan(a) || !(a == a))

#endif // HPGMP_HPP
