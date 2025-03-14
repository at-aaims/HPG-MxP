
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
#if !defined(HPGMP_NO_MPI) && !(defined(HPGMP_WITH_CUDA) || defined(HPGMP_WITH_HIP))

#error "ExchangeHalo GPU version does not work!"

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


#if defined(HPGMP_WITH_HIP)
__global__ void sHaloGather(int totalToBeSent, float *d_x, float *d_sendBuffer, int *d_elementsToSend) {
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i < totalToBeSent) {
    d_sendBuffer[i] = d_x[d_elementsToSend[i]];
  }
}
__global__ void dHaloGather(int totalToBeSent, double *d_x, double *d_sendBuffer, int *d_elementsToSend) {
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i < totalToBeSent) {
    d_sendBuffer[i] = d_x[d_elementsToSend[i]];
  }
}
#endif

template<class SparseMatrix_type, class Vector_type>
void ExchangeHalo_ref(const SparseMatrix_type & A, Vector_type & x) {
  typedef typename SparseMatrix_type::scalar_type scalar_type;
  MPI_Datatype MPI_SCALAR_TYPE = MpiTypeTraits<scalar_type>::getType ();

  // Extract Matrix pieces
  const local_int_t localNumberOfRows = A.localNumberOfRows;
  const local_int_t localNumberOfCols = A.localNumberOfColumns;
  const int num_neighbors = A.numberOfSendNeighbors;
  const local_int_t * receiveLength = A.receiveLength;
  const local_int_t * sendLength = A.sendLength;
  const int * neighbors = A.neighbors;
  scalar_type * sendBuffer = A.sendBuffer;
  const local_int_t totalToBeSent = A.totalToBeSent;
  const local_int_t * elementsToSend = A.elementsToSend;

  scalar_type * const xv = x.values;
  scalar_type * const d_xv = x.d_values;

  int size, rank; // Number of MPI processes, My process ID
  MPI_Comm_size(A.comm, &size);
  MPI_Comm_rank(A.comm, &rank);
  if (size == 1) return;

  //
  //  first post receives, these are immediate receives
  //  Do not wait for result to come, will do that at the
  //  wait call below.
  //

  const int MPI_MY_TAG = 99;

  MPI_Request * request = new MPI_Request[num_neighbors];

  //
  // Externals are at end of locals
  //
#ifdef HPGMP_USE_GPU_AWARE_MPI
  scalar_type * x_external = (scalar_type *) d_xv + localNumberOfRows;
#else
#error "Require GPU-aware MPI!"
  scalar_type * x_external = (scalar_type *) xv + localNumberOfRows;
#endif

  // Post receives first
  // TODO: Thread this loop
  double t0 = 0.0;
  TICK();
  for (int i = 0; i < num_neighbors; i++) {
    local_int_t n_recv = receiveLength[i];
    MPI_Irecv(x_external, n_recv, MPI_SCALAR_TYPE, neighbors[i], MPI_MY_TAG, A.comm, request+i);
    x_external += n_recv;
  }
  double time2 = 0.0;
  TOCK(time2);


  //
  // Fill up send buffer
  //
  TICK();
#if defined(HPGMP_WITH_HIP) // Only with HIP for now
  scalar_type * d_sendBuffer = A.d_sendBuffer;

  int num_threads = (totalToBeSent < 256 ? totalToBeSent : 256);
  int num_blocks = (totalToBeSent+num_threads-1)/num_threads;
  #if defined(HPGMP_WITH_HIP)
  dim3 blocks(num_blocks, 1, 1);
  dim3 threads(num_threads, 1, 1);
  if (std::is_same<scalar_type, float>::value) {
    hipLaunchKernelGGL(sHaloGather, //Kernel name (__global__ void function)
      blocks,        //Grid dimensions
      threads,       //Block dimensions
      0,             //Bytes of dynamic LDS space (ignore for now)
      0,             //Stream (0=NULL stream)
      totalToBeSent, (float*)d_xv, (float*)d_sendBuffer, A.d_elementsToSend); //Kernel arguments
  } else if (std::is_same<scalar_type, double>::value) {
    hipLaunchKernelGGL(dHaloGather, //Kernel name (__global__ void function)
      blocks,        //Grid dimensions
      threads,       //Block dimensions
      0,             //Bytes of dynamic LDS space (ignore for now)
      0,             //Stream (0=NULL stream)
      totalToBeSent, (double*)d_xv, (double*)d_sendBuffer, A.d_elementsToSend); //Kernel arguments
  }
  #ifdef HPGMP_USE_GPU_AWARE_MPI
  hipDeviceSynchronize();
  #else
  if (hipSuccess != hipMemcpy(sendBuffer, A.d_sendBuffer, totalToBeSent*sizeof(scalar_type), hipMemcpyDeviceToHost)) {
    printf( " Failed to memcpy d_y\n" );
  }
  #endif
  #endif
#else
  // Copy local part of X to HOST CPU
  #if defined(HPGMP_WITH_CUDA)
  if (cudaSuccess != cudaMemcpy(xv, d_xv, localNumberOfRows*sizeof(scalar_type), cudaMemcpyDeviceToHost)) {
    printf( " Failed to memcpy d_y\n" );
  }
  #else
  if (hipSuccess != hipMemcpy(xv, d_xv, localNumberOfRows*sizeof(scalar_type), hipMemcpyDeviceToHost)) {
    printf( " Failed to memcpy d_y\n" );
  }
  #endif

  // TODO: Thread this loop
  for (local_int_t i=0; i<totalToBeSent; i++) sendBuffer[i] = xv[elementsToSend[i]];
#endif
  double time1 = 0.0;
  TOCK(time1);

  //
  // Send to each neighbor
  //
  TICK();
  // TODO: Thread this loop
  for (int i = 0; i < num_neighbors; i++) {
    local_int_t n_send = sendLength[i];
    #if defined(HPGMP_USE_GPU_AWARE_MPI) & defined(HPGMP_WITH_HIP)
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
  // TODO: Thread this loop
  for (int i = 0; i < num_neighbors; i++) {
    if ( MPI_Wait(request+i, &status) ) {
      std::exit(-1); // TODO: have better error exit
    }
  }
  TOCK(time2);

  #if !defined(HPGMP_USE_GPU_AWARE_MPI) | !defined(HPGMP_WITH_HIP)
  TICK();
  #if defined(HPGMP_WITH_CUDA)
  if (cudaSuccess != cudaMemcpy(&d_xv[localNumberOfRows], &xv[localNumberOfRows], (localNumberOfCols-localNumberOfRows)*sizeof(scalar_type), cudaMemcpyHostToDevice)) {
    printf( " Failed to memcpy d_y\n" );
  }
  #elif defined(HPGMP_WITH_HIP)
  if (hipSuccess != hipMemcpy(&d_xv[localNumberOfRows], &xv[localNumberOfRows], (localNumberOfCols-localNumberOfRows)*sizeof(scalar_type), hipMemcpyHostToDevice)) {
    printf( " Failed to memcpy d_y\n" );
  }
  #endif
  TOCK(time1);
  #endif

  x.time1 = time1; x.time2 = time2;
  delete [] request;

  return;
}


/* --------------- *
 * specializations *
 * --------------- */

template
void ExchangeHalo_ref< SparseMatrix<double>, Vector<double> >(SparseMatrix<double> const&, Vector<double>&);

template
void ExchangeHalo_ref< SparseMatrix<float>, Vector<float> >(SparseMatrix<float> const&, Vector<float>&);

#endif // ifndef HPGMP_NO_MPI
