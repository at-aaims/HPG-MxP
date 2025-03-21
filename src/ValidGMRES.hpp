
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
 @file ValidGMRES.hpp

 HPGMRES data structure
 */

#ifndef VALIDGMRES_HPP
#define VALIDGMRES_HPP

#include "hpgmp.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "GMRESData.hpp"
#include "device_ctx.hpp"

template<class TestGMRESData_type, class scalar_type, class scalar_type2, class project_type = scalar_type2>
int ValidGMRES(int argc, char **argv, comm_type comm, DeviceCtx *dctx, int numberOfMgLevels, bool verbose, TestGMRESData_type & testcg_data);

#endif  // BENCHGMRES_HPP

