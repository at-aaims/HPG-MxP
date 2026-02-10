
#include "matrix_base.hpp"

#include "SparseMatrix.hpp"

template<typename scalar>
DistMatrixBase::DistMatrixBase(const SparseMatrix<scalar>& A)
    : comm_{A.comm},
      dctx_{A.dctx},
      geom_{A.geom},
      local_nrows_{A.localNumberOfRows},
      local_ncols_{A.localNumberOfColumns},
      numberOfSendNeighbors_{A.numberOfSendNeighbors},
      totalToBeSent_{A.totalToBeSent},
      elementsToSend_{static_cast<local_int_t*>(
          dctx_->device_alloc(totalToBeSent_ * sizeof(local_int_t)))},
      neighbors_{A.neighbors},
      receiveLength_{A.receiveLength},
      sendLength_{A.sendLength},
      halo_row_ind_{static_cast<local_int_t*>(
          dctx_->device_alloc(totalToBeSent_ * sizeof(local_int_t)))},
      n_independent_sets_{A.nblocks},
      ind_perm_{A.perm},
      ind_sizes_{A.sizes},
      ind_offsets_{A.offsets},
      sendBuffer_{dctx_->pinned_host_alloc(totalToBeSent_ * sizeof(scalar))},
      d_sendBuffer_{dctx_->device_alloc(totalToBeSent_ * sizeof(scalar))}
{
    dctx_->copy_host_to_device_sync(elementsToSend_, A.elementsToSend,
                                    totalToBeSent_ * sizeof(local_int_t));
}

template DistMatrixBase::DistMatrixBase(const SparseMatrix<double>& A);
template DistMatrixBase::DistMatrixBase(const SparseMatrix<float>& A);

DistMatrixBase::~DistMatrixBase()
{
    dctx_->pinned_host_free(sendBuffer_);

    dctx_->device_free(halo_row_ind_);
    dctx_->device_free(elementsToSend_);
    dctx_->device_free(d_sendBuffer_);
}
