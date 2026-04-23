
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
 * @file hpgmp.hpp
 * @copyright Modifications (c) 2025 Oak Ridge National Laboratory.
 *
 * HPGMP parameters and helper functions.
 */

#ifndef HPGMP_HPP
#define HPGMP_HPP

#include <fstream>
#include <array>

/*!
  This defines the type for integers that have local subdomain dimension.

  Define as "long long" when local problem dimension is > 2^31
*/
typedef int local_int_t;
//typedef long long local_int_t;

/*!
  This defines the type for integers that have global dimension

  Define as "long long" when global problem dimension is > 2^31
*/
#ifdef HPGMP_NO_LONG_LONG
typedef int global_int_t;
#else
typedef long long global_int_t;
#endif

// This macro should be defined if the global_int_t is not long long
// in order to stop complaints from non-C++11 compliant compilers.
//#define HPGMP_NO_LONG_LONG


extern std::ofstream HPGMP_fout;
extern std::ofstream HPGMP_vout;

struct HPGMP_Params_STRUCT
{
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

/// Type of sparse matrix format to use for benchmark
enum class sp_matrix_format_t {
    mcsr,
    ell
};

/// Type of matrix ordering to use.
enum class ordering_t {
    lex,
    indep_set
};

/// Format types
enum class prec_format_t {
    fp64,
    fp32,
    fp16
};

/// Type of solver validation
enum class validation_t {
    standard,
    fullscale
};

std::string get_string(validation_t val_type);

/// Available types of program run
enum class run_t {
    benchmark, ///< Official benchmark mode (with standard or fullscale validation)
    benchmark_no_ref, ///< Official benchmark mode without timed reference run
    standalone_ref, ///< Only double precision GMRES without validation
    standalone_mxp ///< Only mixed precision GMRES-IR without validation
};

std::string get_string(run_t run_type);

/// Algorithm and data structure options
struct hpgmp_options
{
    sp_matrix_format_t sp_mat_format;
    ordering_t gs_ordering;
};

struct HPGMP_gen_opts
{
    validation_t validation_type;
    run_t run_type;
    double amp_tol;
};

/*!
  HPGMP_Params is a shorthand for HPGMP_Params_STRUCT
 */
typedef HPGMP_Params_STRUCT HPGMP_Params;
#ifdef HPGMP_NO_MPI
typedef int comm_type;
#define MPI_Abort(comm, errorcode) abort();
#define MPI_Barrier(comm) ;
#else
#include <mpi.h>
typedef MPI_Comm comm_type;
#endif

int HPGMP_Init_Params(const char* title, int* argc_p, char*** argv_p, HPGMP_Params& params, comm_type comm);
int HPGMP_Init_Params(int* argc_p, char*** argv_p, HPGMP_Params& params, comm_type comm);
HPGMP_gen_opts HPGMP_Init(int* argc_p, char*** argv_p);
int HPGMP_Finalize(void);

#define IS_NAN(a) (std::isinf(a) || std::isnan(a) || !(a == a))

#ifdef HPGMP_VERBOSE
#define HPGMP_VERBOSE_PRINT(_msg) \
    printf(_msg);                 \
    printf("\n");                 \
    fflush(stdout)
#else
#define HPGMP_VERBOSE_PRINT(_msg)
#endif

#endif // HPGMP_HPP
