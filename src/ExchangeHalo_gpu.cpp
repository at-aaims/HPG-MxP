
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
 @file ExchangeHalo.cpp

 HPGMP routine
 */

// Compile this routine only if running with MPI
#if !defined(HPGMP_NO_MPI) && (defined(HPGMP_WITH_CUDA) || defined(HPGMP_WITH_HIP))

#include <mpi.h>
#include "Utils_MPI.hpp"
#include "Geometry.hpp"
#include "ExchangeHalo.hpp"
#include "mytimer.hpp"
#include <cstdlib>

/*!
  Communicates data that is at the border of the part of the domain assigned to this processor.

  @param[in]    A The known system matrix
  @param[inout] x On entry: the local vector entries followed by entries to be communicated; on exit: the vector with non-local entries updated by other processors
 */


template<typename scalar>
__global__ void haloGather(const int totalToBeSent, const scalar* const d_x, scalar* const d_sendBuffer, const int* const d_elementsToSend)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < totalToBeSent) {
        d_sendBuffer[i] = d_x[d_elementsToSend[i]];
    }
}

template<class SparseMatrix_type, class Vector_type>
void ExchangeHalo_ref(const SparseMatrix_type& A, Vector_type& x)
{

    HPGMP_RANGE_PUSH(__FUNCTION__);

    typedef typename SparseMatrix_type::scalar_type scalar_type;
    MPI_Datatype MPI_SCALAR_TYPE = MpiTypeTraits<scalar_type>::getType();

    // Extract Matrix pieces
    const local_int_t localNumberOfRows = A.localNumberOfRows;
    const local_int_t localNumberOfCols = A.localNumberOfColumns;
    const int num_neighbors             = A.numberOfSendNeighbors;
    const local_int_t* receiveLength    = A.receiveLength;
    const local_int_t* sendLength       = A.sendLength;
    const int* neighbors                = A.neighbors;
    scalar_type* sendBuffer             = A.sendBuffer;
    const local_int_t totalToBeSent     = A.totalToBeSent;

#ifndef HPGMP_USE_GPU_AWARE_MPI
    scalar_type* const xv = x.values();
#endif
    scalar_type* const d_xv = x.d_values();

    int size, rank; // Number of MPI processes, My process ID
    MPI_Comm_size(A.comm, &size);
    MPI_Comm_rank(A.comm, &rank);
    if (size == 1) {
        HPGMP_RANGE_POP(__FUNCTION__);
        return;
    }

    //
    //  first post receives, these are immediate receives
    //  Do not wait for result to come, will do that at the
    //  wait call below.
    //

    const int MPI_MY_TAG = 99;

    MPI_Request* request = new MPI_Request[num_neighbors];

    //
    // Externals are at end of locals
    //
#ifdef HPGMP_USE_GPU_AWARE_MPI
    scalar_type* x_external = (scalar_type*)d_xv + localNumberOfRows;
#else
    scalar_type* x_external = (scalar_type*)xv + localNumberOfRows;
#endif

    auto dctx        = x.get_device_context();
    auto halo_stream = dctx->get_halo_stream();

    // Post receives first
    // TODO: Thread this loop
    double t0 = 0.0;
    TICK();
    for (int i = 0; i < num_neighbors; i++) {
        local_int_t n_recv = receiveLength[i];
        MPI_Irecv(x_external, n_recv, MPI_SCALAR_TYPE, neighbors[i], MPI_MY_TAG, A.comm, request + i);
        x_external += n_recv;
    }
    double time2 = 0.0;
    TOCK(time2);


    //
    // Fill up send buffer
    //
    TICK();
    scalar_type* d_sendBuffer = A.d_sendBuffer;

    const int num_threads = (totalToBeSent < 256 ? totalToBeSent : 256);
    const int num_blocks  = (totalToBeSent + num_threads - 1) / num_threads;
    haloGather<<<num_blocks, num_threads, 0, halo_stream>>>(
        totalToBeSent, d_xv, d_sendBuffer, A.d_elementsToSend);

    dctx->synchronize_halo_stream();

#if !defined(HPGMP_USE_GPU_AWARE_MPI)
    if (hipSuccess != hipMemcpy(sendBuffer, A.d_sendBuffer, totalToBeSent * sizeof(scalar_type),
                                hipMemcpyDeviceToHost)) {
        printf(" Failed to memcpy d_y\n");
    }
#endif

    double time1 = 0.0;
    TOCK(time1);

    //
    // Send to each neighbor
    //
    TICK();
    for (int i = 0; i < num_neighbors; i++) {
        local_int_t n_send = sendLength[i];
#if defined(HPGMP_USE_GPU_AWARE_MPI)
        MPI_Send(d_sendBuffer, n_send, MPI_SCALAR_TYPE, neighbors[i], MPI_MY_TAG, A.comm);
#else
        MPI_Send(sendBuffer, n_send, MPI_SCALAR_TYPE, neighbors[i], MPI_MY_TAG, A.comm);
#endif
        sendBuffer += n_send;
    }

    //
    // Complete the reads issued above
    //

    MPI_Status status;
    for (int i = 0; i < num_neighbors; i++) {
        if (MPI_Wait(request + i, &status)) {
            throw std::runtime_error("Sync exchangehalo wait did not complete!");
        }
    }
    TOCK(time2);

#if !defined(HPGMP_USE_GPU_AWARE_MPI)
    // copy received data to GPU
    TICK();
#if defined(HPGMP_WITH_CUDA)
    if (cudaSuccess != cudaMemcpy(d_xv + localNumberOfRows, &xv[localNumberOfRows], (localNumberOfCols - localNumberOfRows) * sizeof(scalar_type), cudaMemcpyHostToDevice)) {
        printf(" Failed to memcpy d_y\n");
    }
#elif defined(HPGMP_WITH_HIP)
    if (hipSuccess != hipMemcpy(d_xv + localNumberOfRows, &xv[localNumberOfRows], (localNumberOfCols - localNumberOfRows) * sizeof(scalar_type), hipMemcpyHostToDevice)) {
        printf(" Failed to memcpy d_y\n");
    }
#endif
    TOCK(time1);
#endif

    x.time1 = time1;
    x.time2 = time2;
    delete[] request;

    HPGMP_RANGE_POP(__FUNCTION__);

    return;
}


/* --------------- *
 * specializations *
 * --------------- */

template void ExchangeHalo_ref< SparseMatrix<double>, Vector<double> >(
    SparseMatrix<double> const&, Vector<double>&);

template void ExchangeHalo_ref< SparseMatrix<float>, Vector<float> >(
    SparseMatrix<float> const&, Vector<float>&);

#endif // ifndef HPGMP_NO_MPI
