/**
 * @file OptimizeProblem.cpp
 * @copyright (c) 2025 Oak Ridge National Laboratory.
 */

#include "OptimizeProblem.hpp"

#include "DataTypes.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "GMRESData.hpp"

template<typename local_scalar_type, typename halo_scalar_t, typename GMRESData_type, typename vec_scalar_t>
int OptimizeProblemELL(SparseMatrix<local_scalar_type, halo_scalar_t>& A, GMRESData_type& data,
                       Vector<vec_scalar_t>& b, Vector<vec_scalar_t>& x, Vector<vec_scalar_t>& xexact);

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
#endif
    return 0;
}

template int OptimizeProblem(
    SparseMatrix<double>&, GMRESData<double, double, double>&, Vector<double>&, Vector<double>&,
    Vector<double>&);
template int OptimizeProblem(
    SparseMatrix<float>&, GMRESData<float, float, float>&, Vector<float>&, Vector<float>&,
    Vector<float>&);
template int OptimizeProblem(
    SparseMatrix<float>&, GMRESData<double, double, double>&, Vector<double>&, Vector<double>&,
    Vector<double>&);
#ifdef HPGMP_WITH_GINKGO_AMP
template int OptimizeProblem(
    SparseMatrix<double, float>&, GMRESData<double, double, double>&, Vector<double>&, Vector<double>&,
    Vector<double>&);
#endif
