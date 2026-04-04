
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
 @file Vector.hpp

 HPGMP routine
 */

#ifndef MULTIVECTOR_HPP
#define MULTIVECTOR_HPP

#include "DataTypes.hpp"
#include "device_ctx.hpp"
#include "Vector.hpp"

/**
 * Several vectors of the same length, stored as columns of a dense matrix in column-major layout.
 */
template<typename scalar_t>
class MultiVector
{
public:
    typedef scalar_t scalar_type;

    /// Create empty multivector
    MultiVector();

    /// Create a new multivector that owns its data.
    MultiVector(local_int_t localLength, local_int_t n_vectors, comm_type comm, DeviceCtx* dev_ctx);

    /// Create a (mutable) view into another multivector; this does not own the data.
    MultiVector(local_int_t localLength, local_int_t n_vectors, comm_type comm, DeviceCtx* dev_ctx,
                scalar_t* values, scalar_t* d_values);

    /// Finalize the multivector automatically after use.
    ~MultiVector();

    comm_type get_comm() const { return comm_; }

    dev_blas_ctx get_blas_handle() const { return dctx_->get_blas_handle(); }

    /*!
   * Initializes the vectors.
   * 
   * @param[in] localLength Length of local portion of input vector
   * @param[in] n           Number of columns
   */
    void initialize(local_int_t localLength, local_int_t n_vectors, comm_type comm, DeviceCtx* dev_ctx);

    /// Initializes the vectors as a view into another multivector
    void initialize_view(local_int_t localLength, local_int_t n_vectors, comm_type comm,
                         DeviceCtx* dev_ctx, scalar_t* values, scalar_t* d_values);

    /// Returns a view of one of the vectors
    Vector<scalar_t> get_vector(local_int_t j);

    /**
   * Returns a view of a few contiguous vectors
   *
   * @param j1  The first column vector to include in the view.
   * @param j2  The last column vector to include in the view.
   *            Note that column j2 is also included!
   */
    MultiVector<scalar_t> get_multi_vector(local_int_t j1, local_int_t j2);

    const scalar_type* d_values() const { return d_values_; }

    scalar_type* d_values() { return d_values_; }

    const scalar_type* values() const { return values_; }

    /// Updates the host copy of the multivector from device data.
    void update_host_mirror() const;

    /// Updates the device data of the multivector from the host buffer's data.
    void update_device_data() const;

    void fill_zero();

private:
    local_int_t n_           = 0; //!< number of vectors
    local_int_t localLength_ = 0; //!< length of local portion of the vector
    scalar_t* values_        = nullptr; //!< array of values

    //#if defined(HPGMP_WITH_CUDA) | defined(HPGMP_WITH_HIP)
    scalar_t* d_values_ = nullptr; //!< array of values
    //#endif

    /// communicator
    comm_type comm_;

    /// Device context of device on which vector resides.
    DeviceCtx* dctx_;

    bool is_view_ = false;

    /*!
   This is for storing optimized data structures created in OptimizeProblem and
   used inside optimized ComputeSPMV().
   */
    void* optimizationData_ = nullptr;
};

#endif // MULTIVECTOR_HPP
