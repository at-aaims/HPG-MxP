/**
 * @file OptimizeProblem.cpp
 * @copyright (c) 2025 Oak Ridge National Laboratory.
 */

#include "OptimizeProblem.hpp"

#include "DataTypes.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "GMRESData.hpp"

template<typename mat_scalar, typename solver_scalar, typename vec_scalar>
int OptimizeProblemELL(SparseMatrix<mat_scalar>& A, GMRESData<solver_scalar>& data,
                       Vector<vec_scalar>& b, Vector<vec_scalar>& x, Vector<vec_scalar>& xexact);

template<typename SparseMatrix_type, typename GMRESData_type, typename Vector_type>
int OptimizeProblem_ref(SparseMatrix_type& A, GMRESData_type& data,
                        Vector_type& b, Vector_type& x, Vector_type& xexact);

template<class SparseMatrix_type, class GMRESData_type, class Vector_type>
int OptimizeProblem(SparseMatrix_type& A, GMRESData_type& data, Vector_type& b, Vector_type& x,
                    Vector_type& xexact)
{
#ifdef HPGMP_REFERENCE
    OptimizeProblem_ref(A, data, b, x, xexact);
#else
    OptimizeProblemELL(A, data, b, x, xexact);
#ifdef HPGMP_WITH_GINKGO
    // Generate Ginkgo AMP matrix, given ELL
#endif
#endif
    return 0;
}

template int OptimizeProblem(
    SparseMatrix<double>&, GMRESData<double>&, Vector<double>&, Vector<double>&,
    Vector<double>&);
template int OptimizeProblem(
    SparseMatrix<float>&, GMRESData<float>&, Vector<float>&, Vector<float>&,
    Vector<float>&);
template int OptimizeProblem(
    SparseMatrix<float>&, GMRESData<double>&, Vector<double>&, Vector<double>&,
    Vector<double>&);
