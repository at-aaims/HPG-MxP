
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

#ifndef HPGMP_VECTOR_HPP
#define HPGMP_VECTOR_HPP

#include <fstream>
#include <cassert>
#include <cstdlib>
#include <stdexcept>

#ifdef HPGMP_WITH_BLAS
#include "cblas.h"
#endif

#ifndef HPGMP_NO_MPI
#include <vector>
#include <mpi.h>
#endif

#include "device_ctx.hpp"
#include "hpgmp.hpp"
#include "matrix_base.hpp"

template<typename scalar_t = double>
class Vector
{
public:
    typedef scalar_t scalar_type;

    /** Creates uninitialized vector.
   *
   * @sa initialize
   */
    Vector();

    /** Creates a vector.
   *
   * @param local_length  Length of local part of the vector.
   * @param comm  MPI communicator associated with the distributed vector.
   * @param device_ctx  Context of the device (accelerator/CPU) on which this vector resides.
   */
    Vector(local_int_t localLength, comm_type comm, DeviceCtx* dev_ctx);

    /** Creates a vector view.
   *
   * @param local_length  Length of local part of the vector.
   * @param comm  MPI communicator associated with the distributed vector.
   * @param device_ctx  Context of the device (accelerator/CPU) on which this vector resides.
   */
    Vector(local_int_t localLength, comm_type comm, DeviceCtx* dev_ctx, scalar_t* values, scalar_t* d_values);

    ~Vector();

    /// Initialize the vector.
    void initialize(local_int_t localLength, comm_type comm, DeviceCtx* dev_ctx);

    /// Initialize a non-managing view into a different entity.
    void initialize_view(local_int_t localLength, comm_type comm, DeviceCtx* dev_ctx,
                         scalar_t* values, scalar_t* d_values);

    scalar_t* values() { return values_; }
    scalar_t* d_values() { return d_values_; }
    const scalar_t* values() const { return values_; }
    const scalar_t* d_values() const { return d_values_; }
    local_int_t local_length() const { return localLength_; }

    DeviceCtx* get_device_context() const { return dctx_; }
    comm_type get_comm() const { return comm_; }

    dev_blas_ctx get_blas_handle() const { return dctx_->get_blas_handle(); }

    /// Updates the host copy of the vector from device data.
    void update_host_mirror() const;

    /// Updates the device data of the vector from the host buffer's data.
    void update_device_data() const;

    /** @brief Begin halo update for operations like SpMV and SpTRSV.
   *
   * For a non-blocking build, this performs asynchronous packing of the send buffer
   * on the halo stream. It also inserts a wait event on the (interior) compute stream.
   *
   * For a blocking build, it packs the send buffer, issues MPI receives and sends,
   * waits for these to complete, and if necessary, copies the received buffer to GPU.
   *
   * Builds are blocking if HPGMP_ONLY_BLOCKING_COMMS is set.
   */
    void update_halos_pack_send_buffer(const DistMatrixBase* mat) const;

    /// (Non-blocking builds only) Synchronize halos tream and issue asynchronous send and receives.
    void update_halos_send_receive(const DistMatrixBase* mat) const;

    /// (Non-blocking builds only) Wait for the communications and send data to GPU if necessary.
    void update_halos_finalize(const DistMatrixBase* mat) const;

    // Some operations

    /// Fill vector with zero values
    void fill_zero();

    /// Fills vector with random values between 1 and 2.
    void fill_random();

    /// Scales the vector by multiplying with a scalar.
    void scale(scalar_type value);

    /** @brief Permutes a vector based on given indices.
   *
   * @param perm  Local indices such that v_new[perm[i]] = v_old[i].
   *              We assume it has the same length as this vector.
   *
   * @warning Calling this function invalidates outstanding outside pointers
   *          returned by d_values().
   */
    void permute(const local_int_t* perm);

    mutable double time1_{}, time2_{};
    mutable double time3_{}, time4_{};

private:
    local_int_t localLength_ = 0; //!< length of local portion of the vector

    /// communicator
    comm_type comm_;

    /// Device context for this vector
    DeviceCtx* dctx_;

    scalar_t* values_ = nullptr; //!< array of values

    scalar_t* d_values_ = nullptr; //!< array of values
    bool is_view_       = false;

#ifndef HPGMP_NO_MPI
    event_t send_gather_;
    mutable std::vector<MPI_Request> send_reqs_;
    mutable std::vector<MPI_Request> recv_reqs_;
    mutable bool halos_buffer_packed_ = false;
#endif

    /*!
   This is for storing optimized data structures created in OptimizeProblem and
   used inside optimized ComputeSPMV().
   */
    void* optimizationData = nullptr;

    mutable double t0_;
};

/*!
 * Copy input vector to output vector.
 *
 * @param[in] v Input vector
 * @param[in] w Output vector
 */
template<typename scalar_src, typename scalar_dst>
void CopyVector(const Vector<scalar_src>& v, Vector<scalar_dst>& w);

#endif // VECTOR_HPP
