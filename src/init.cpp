
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

#ifndef HPGMP_NO_MPI
#include <mpi.h>
#endif

#ifndef HPGMP_NO_OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
const char* NULLDEVICE = "nul";
#else
const char* NULLDEVICE = "/dev/null";
#endif

#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "hpgmp.hpp"
#include "DataTypes.hpp"
#include "Utils_MPI.hpp"

#include "ReadHpgmpDat.hpp"


std::ofstream HPGMP_fout; //!< output file stream for logging activities during HPGMP run
std::ofstream HPGMP_vout; //!< output file stream for verbose logging activities during HPGMP run

static int
startswith(const char* s, const char* prefix)
{
    size_t n = strlen(prefix);
    if (strncmp(s, prefix, n))
        return 0;
    return 1;
}

// Reading configuration file for backward compatibility
// @todo Remove this eventually, and only support command line
#ifdef HPGMP_WITH_GINKGO_AMP
int read_amp_config(double& tol)
{
    std::ifstream amp_config_file("amp_config.txt");

    if (amp_config_file.is_open()) {
        amp_config_file >> tol;
        amp_config_file.close();
    }

    return 0;
}
#endif

HPGMP_gen_opts
HPGMP_Init(int* argc_p, char*** argv_p)
{
    HPGMP_gen_opts opts;
    const int argc = *argc_p;
    char** argv    = *argv_p;
    // Options to be read
    constexpr int nparams                          = 3;
    const std::array<std::string, nparams> cparams = {"--validation_type", "--run_type", "--amp_tol"};
    std::array<std::string, nparams> values;
    // Default values
    values[0] = "standard";
    values[1] = "benchmark";
    values[2] = "1e-8";

    // Read cmd line args
    for (int i = 1; i <= argc && argv[i]; ++i) {
        std::stringstream ss(argv[i]);
        std::string prefix;
        if (std::getline(ss, prefix, '=')) {
            for (int j = 0; j < nparams; ++j) {
                if (prefix == cparams[j]) {
                    if (!std::getline(ss, values[j])) {
                        throw std::runtime_error("Could not read cmd line value!");
                    }
                }
            }
        } else {
            throw std::runtime_error("Could not read cmd line option prefix!");
        }
    }

    // Set validation_type in opts
    if (values[0] == "standard") {
        opts.validation_type = validation_t::standard;
    } else if (values[0] == "fullscale") {
        opts.validation_type = validation_t::fullscale;
    } else {
        throw std::runtime_error("Invalid value for validation_type!");
    }

    // Set run_type in opts
    if (values[1] == "benchmark") {
        opts.run_type = run_t::benchmark;
    } else if (values[1] == "benchmark_no_ref") {
        opts.run_type = run_t::benchmark_no_ref;
    } else if (values[1] == "standalone_ref") {
        opts.run_type = run_t::standalone_ref;
    } else if (values[1] == "standalone_mxp") {
        opts.run_type = run_t::standalone_mxp;
    } else {
        throw std::runtime_error("Invalid value for run_type!");
    }

    // Set amp_tol in opts
    opts.amp_tol = std::stod(values[2]);

#ifdef HPGMP_WITH_GINKGO_AMP
    // Reading configuration file for backward compatibility.
    // This overrides the command line argument if the file is provided.
    // @todo Remove this eventually, and only support command line
    read_amp_config(opts.amp_tol);
#endif

    return opts;
}


/*!
  Initializes an HPGMP run by obtaining problem parameters (from a file or
  command line) and then broadcasts them to all nodes. It also initializes
  login I/O streams that are used throughout the HPGMP run. Only MPI rank 0
  performs I/O operations.

  The function assumes that MPI has already been initialized for MPI runs.

  @param[in] argc_p the pointer to the "argc" parameter passed to the main() function
  @param[in] argv_p the pointer to the "argv" parameter passed to the main() function
  @param[out] params the reference to the data structures that is filled the basic parameters of the run

  @return returns 0 upon success and non-zero otherwise

  @see HPGMP_Finalize
*/
int HPGMP_Init_Params(const char* title, int* argc_p, char*** argv_p, HPGMP_Params& params, comm_type comm)
{
    const int argc = *argc_p;
    char** argv    = *argv_p;
    char fname[80];
    int i, j, *iparams;
    char cparams[][7] = {"--nx=", "--ny=", "--nz=", "--rt=", "--pz=", "--zl=", "--zu=", "--npx=", "--npy=", "--npz="};
    time_t rawtime;
    tm* ptm;
    const int nparams = (sizeof cparams) / (sizeof cparams[0]);
#ifndef HPGMP_NO_MPI
    bool broadcastParams = false; // Make true if parameters read from file.
#endif
    iparams = (int*)malloc(sizeof(int) * nparams);

    // Initialize iparams
    for (i = 0; i < nparams; ++i) iparams[i] = 0;

    /* for sequential and some MPI implementations it's OK to read first three args */
    for (i = 0; i < nparams; ++i)
        if (argc <= i + 1 || sscanf(argv[i + 1], "%d", iparams + i) != 1 || iparams[i] < 10) iparams[i] = 0;

    /* for some MPI environments, command line arguments may get complicated so we need a prefix */
    for (i = 1; i <= argc && argv[i]; ++i)
        for (j = 0; j < nparams; ++j)
            if (startswith(argv[i], cparams[j]))
                if (sscanf(argv[i] + strlen(cparams[j]), "%d", iparams + j) != 1)
                    iparams[j] = 0;

    // Check if --rt was specified on the command line
    int* rt = iparams + 3; // Assume runtime was not specified and will be read from the hpcg.dat file
    if (iparams[3]) rt = 0; // If --rt was specified, we already have the runtime, so don't read it from file
    if (!iparams[0] && !iparams[1] && !iparams[2]) { /* no geometry arguments on the command line */
        ReadHpgmpDat(iparams, rt, iparams + 7);
#ifndef HPGMP_NO_MPI
        broadcastParams = true;
#endif
    }

    // Check for small or unspecified nx, ny, nz values
    // If any dimension is less than 16, make it the max over the other two dimensions, or 16, whichever is largest
    for (i = 0; i < 3; ++i) {
        if (iparams[i] < 16)
            for (j = 1; j <= 2; ++j)
                if (iparams[(i + j) % 3] > iparams[i])
                    iparams[i] = iparams[(i + j) % 3];
        if (iparams[i] < 16)
            iparams[i] = 16;
    }

// Broadcast values of iparams to all MPI processes
#ifndef HPGMP_NO_MPI
    if (broadcastParams) {
        MPI_Bcast(iparams, nparams, MPI_INT, 0, comm);
    }
#endif

    params.nx = iparams[0];
    params.ny = iparams[1];
    params.nz = iparams[2];

    params.runningTime = iparams[3];
    params.pz          = iparams[4];
    params.zl          = iparams[5];
    params.zu          = iparams[6];

    params.npx = iparams[7];
    params.npy = iparams[8];
    params.npz = iparams[9];

#ifndef HPGMP_NO_MPI
    MPI_Comm_rank(comm, &params.comm_rank);
    MPI_Comm_size(comm, &params.comm_size);
#else
    params.comm_rank = 0;
    params.comm_size = 1;
#endif

#ifdef HPGMP_NO_OPENMP
    params.numThreads = 1;
#else
    // clang-format off
    #pragma omp parallel
    // clang-format on
    params.numThreads = omp_get_num_threads();
#endif
    //  for (i = 0; i < nparams; ++i) std::cout << "rank = "<< params.comm_rank << " iparam["<<i<<"] = " << iparams[i] << "\n";

    time(&rawtime);
    ptm = localtime(&rawtime);
    HPGMP_fout.close();
    if (0 == params.comm_rank) {
        sprintf(fname, "%shpgmp%04d%02d%02dT%02d%02d%02d.txt", title,
                1900 + ptm->tm_year, ptm->tm_mon + 1, ptm->tm_mday, ptm->tm_hour, ptm->tm_min, ptm->tm_sec);
        HPGMP_fout.open(fname);
#if defined(HPGMP_DETAILED_PRINT)
        HPGMP_vout.open(fname);
#else
        HPGMP_vout.open(NULLDEVICE);
#endif
    } else {
#if defined(HPGMP_DEBUG) || defined(HPGMP_DETAILED_DEBUG)
        sprintf(fname, "%shpgmp%04d%02d%02dT%02d%02d%02d_%d.txt", title,
                1900 + ptm->tm_year, ptm->tm_mon + 1, ptm->tm_mday, ptm->tm_hour, ptm->tm_min, ptm->tm_sec, params.comm_rank);
        HPGMP_fout.open(fname);
        if (params.comm_rank == 0) {
            std::cout << title << ": running time = " << params.runningTime << std::endl;
        }
#else
        HPGMP_fout.open(NULLDEVICE);
#endif
    }
    free(iparams);

    return 0;
}

int HPGMP_Init_Params(int* argc_p, char*** argv_p, HPGMP_Params& params, comm_type comm)
{
    return HPGMP_Init_Params("", argc_p, argv_p, params, comm);
}
