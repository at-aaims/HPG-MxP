
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
 @file BenchGMRES.hpp

 HPGMRES data structure
 */

#ifndef BENCHGMRES_HPP
#define BENCHGMRES_HPP

#include "hpgmp.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "device_ctx.hpp"
#include "GMRESData.hpp"

template<class scalar_type, class scalar_type2, class project_type = scalar_type2>
int BenchGMRES(int argc, char **argv, comm_type comm, DeviceCtx *dctx, int numberOfMgLevels,
               bool verbose, bool validation_failure, const HPGMP_gen_opts& gopts,
               TestGMRESData& testcg_data);

#endif  // BENCHGMRES_HPP

