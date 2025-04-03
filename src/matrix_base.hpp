/**
 * @file matrix_base.hpp
 * @author Aditya Kashi
 * @copyright (C) 2025, Oak Ridge National Laboratory.
 */

#ifndef HPGMP_MATRIX_BASE
#define HPGMP_MATRIX_BASE

#include "Geometry.hpp"
#include "device_ctx.hpp"
#include "hpgmp.hpp"

template <typename T>
class SparseMatrix;

/**
 * @brief Base class for distributed sparse matrices.
 *
 * It is envisioned that any (reference or optimized) implementation will inherit this base class.
 * The interface provided by this base class can be used:
 * - by distributed vectors to exchange halos.
 * -
 */
class DistMatrixBase
{
public:
    DistMatrixBase(comm_type comm, DeviceCtx *const dctx, const Geometry *const geom)
        : comm_{comm}, dctx_{dctx}, geom_{geom}
    { }
   
    template <typename scalar> 
    DistMatrixBase(const SparseMatrix<scalar>& A);

    virtual ~DistMatrixBase();

    DeviceCtx *get_device_context() const { return dctx_; }
    comm_type get_comm() const { return comm_; }
    const Geometry *get_geometry() const { return geom_; }

    local_int_t get_local_num_rows() const { return local_nrows_; }
    local_int_t get_local_num_cols() const { return local_ncols_; }

    local_int_t get_num_halo_rows() const { return n_halo_rows_; }
    local_int_t get_total_to_be_sent() const { return totalToBeSent_; }

    int get_num_neighbors() const { return numberOfSendNeighbors_; }

    /// Get list of vector entries to send; residing on device.
    const local_int_t* get_elements_to_send() const { return elementsToSend_; }
    const int* get_neighbors() const { return neighbors_; }
    const local_int_t *get_receive_lengths() const { return receiveLength_; }
    const local_int_t *get_send_lengths() const { return sendLength_; }
    const local_int_t *get_halo_row_indices() const { return halo_row_ind_; }

    void *get_host_send_buffer() const { return sendBuffer_; }
    void *get_device_send_buffer() const { return d_sendBuffer_; }
    const local_int_t* get_reordering_permutation() const { return perm_; }

protected:
    comm_type comm_;
    DeviceCtx *dctx_;
    const Geometry *geom_;
    local_int_t local_nrows_{};
    local_int_t local_ncols_{};
    global_int_t total_nrows_{};
    /// No. of halo points.
    local_int_t n_halo_rows_{};

    // For comms
    /// number of halo points that are external to this process
    //local_int_t numberOfExternalValues_ = 0;
    /// number of neighboring processes that will be sent local data
    int numberOfSendNeighbors_ = 0;
    /// total number of entries to be sent [Should be same as numberOfExternalValues_ ?]
    local_int_t totalToBeSent_ = 0; 
    /// Local elements of a distributed vector that must be sent to neighboring processes (GPU)
    local_int_t * elementsToSend_ = nullptr;
    /// neighboring processes
    int * neighbors_ = nullptr;
    /// lenghts of messages received from neighboring processes (CPU)
    local_int_t * receiveLength_ = nullptr;
    /// lenghts of messages sent to neighboring processes
    local_int_t * sendLength_ = nullptr;
    /// Local indices of halo rows
    local_int_t *halo_row_ind_ = nullptr;
    /// Permutation vector, eg., computer by independent set ordering (device)
    local_int_t *perm_ = nullptr;

    /** Buffer for communicating vectors.
     *
     * Note that these buffers are not considered a logical part of the matrix.
     * They are only meant to be used as temporary storage and not to store semantically important
     * information about this matrix.
     */
    void *sendBuffer_ = nullptr;
    void *d_sendBuffer_ = nullptr;

    //virtual void setup_matrix() = 0;
    //virtual void setup_halo() = 0;
};

#endif
