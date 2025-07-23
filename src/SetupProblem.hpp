
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
 @file GenerateProblem.cpp

 HPGMP routine
 */

#ifndef SETUP_PROBLEM_HPP
#define SETUP_PROBLEM_HPP

#include "device_ctx.hpp"
#include "Geometry.hpp"
#include "GMRESData.hpp"

template<class SparseMatrix_type, class SparseMatrix_type2, class GMRESData_type, class GMRESData_type2, class Vector_type>
void SetupProblem(const char *title, int argc, char **argv, comm_type comm, DeviceCtx *dctx, int numberOfMgLevels, bool verbose, Geometry * geom,
                  SparseMatrix_type & A, GMRESData_type & data, SparseMatrix_type2 & A2, GMRESData_type2 & data2,
                  Vector_type & b, Vector_type & x, TestGMRESData& test_data);

#endif
