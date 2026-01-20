
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
 @file SetupMatrix.cpp

 HPGMP routine
 */

#ifndef SETUP_MATRIX_HPP
#define SETUP_MATRIX_HPP

#ifndef HPGMP_NO_MPI
#include <mpi.h>
#endif

#ifndef HPGMP_NO_OPENMP
#include <omp.h>
#endif

#include "GMRESData.hpp"
#include "Geometry.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "GenerateNonsymProblem.hpp"
#include "GenerateNonsymCoarseProblem.hpp"
#include "SetupHalo.hpp"


/*!
  Routine to generate a sparse matrix, right hand side, initial guess, and exact solution.

  @param[in]  A        The generated system matrix
  @param[inout] b      The newly allocated and generated right hand side vector (if b!=0 on entry)
  @param[inout] x      The newly allocated solution vector with entries set to 0.0 (if x!=0 on entry)
  @param[inout] xexact The newly allocated solution vector with entries set to the exact solution (if the xexact!=0 non-zero on entry)

  @see GenerateGeometry
*/

template<class SparseMatrix_type, class GMRESData_type, class Vector_type>
void SetupMatrix(DeviceCtx* const dctx, int numberOfMgLevels, SparseMatrix_type& A, Geometry* geom,
                 GMRESData_type& data, Vector_type* b, Vector_type* x, Vector_type* xexact,
                 bool init_vect, comm_type comm);
#endif
