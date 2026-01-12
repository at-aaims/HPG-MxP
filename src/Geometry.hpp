
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
 @file Geometry.hpp

 HPGMP data structure for problem geometry
 */

#ifndef GEOMETRY_HPP
#define GEOMETRY_HPP

#include <array>

#include "hpgmp.hpp"

/*!
  This is a data structure to contain all processor geometry information
*/
struct Geometry_STRUCT
{
    int size; //!< Number of MPI processes
    int rank; //!< This process' rank in the range [0 to size - 1]
    int numThreads; //!< This process' number of threads
    local_int_t nx; //!< Number of x-direction grid points for each local subdomain
    local_int_t ny; //!< Number of y-direction grid points for each local subdomain
    local_int_t nz; //!< Number of z-direction grid points for each local subdomain
    int npx; //!< Number of processors in x-direction
    int npy; //!< Number of processors in y-direction
    int npz; //!< Number of processors in z-direction
    int pz; //!< partition ID of z-dimension process that starts the second region of nz values
    int npartz; //!< Number of partitions with varying nz values
    std::array<int, 2> partz_ids;
    std::array<local_int_t, 2> partz_nz;
    int ipx; //!< Current rank's x location in the npx by npy by npz processor grid
    int ipy; //!< Current rank's y location in the npx by npy by npz processor grid
    int ipz; //!< Current rank's z location in the npx by npy by npz processor grid
    global_int_t gnx; //!< Global number of x-direction grid points
    global_int_t gny; //!< Global number of y-direction grid points
    global_int_t gnz; //!< Global number of z-direction grid points
    global_int_t gix0; //!< Base global x index for this rank in the npx by npy by npz processor grid
    global_int_t giy0; //!< Base global y index for this rank in the npx by npy by npz processor grid
    global_int_t giz0; //!< Base global z index for this rank in the npx by npy by npz processor grid
};
typedef struct Geometry_STRUCT Geometry;

/*!
 * Computes the factorization of the total number of processes into a
 * 3-dimensional process grid that is as close as possible to a cube. The
 * quality of the factorization depends on the prime number structure of the
 * total number of processes. It then stores this decompostion together with the
 * parallel parameters of the run in the geometry data structure.

 * @param[in] size  Total number of MPI processes
 * @param[in] rank  This process' rank among other MPI processes
 * @param[in] numThreads  Number of OpenMP threads in this process
 * @param[in] pz  z-dimension processor ID where second zone of nz values start
 * @param[in] nx, ny, nz  Number of grid points for each local block in the x, y, and z dimensions
 *             respectively
 * @param[out] geom  Data structure that will store the above parameters and
 *                   the factoring of total number of processes into three dimensions
*/
void GenerateGeometry(const int size, const int rank, const int numThreads,
                      const int pz, const local_int_t zl, const local_int_t zu,
                      const local_int_t nx, const local_int_t ny, const local_int_t nz,
                      int npx, int npy, int npz,
                      Geometry* const geom);

/*!
  Returns the rank of the MPI process that is assigned the global row index
  given as the input argument.

  @param[in] geom  The description of the problem's geometry.
  @param[in] index The global row index

  @return Returns the MPI rank of the process assigned the row
*/
inline int ComputeRankOfMatrixRow(const Geometry& geom, global_int_t index)
{
    global_int_t gnx = geom.gnx;
    global_int_t gny = geom.gny;

    global_int_t iz = index / (gny * gnx);
    global_int_t iy = (index - iz * gny * gnx) / gnx;
    global_int_t ix = index % gnx;
    // We now permit varying values for nz for any nx-by-ny plane of MPI processes.
    // npartz is the number of different groups of nx-by-ny groups of processes.
    // partz_ids is an array of length npartz where each value indicates the z process of the last process in the ith nx-by-ny group.
    // partz_nz is an array of length npartz containing the value of nz for the ith group.

    //        With no variation, npartz = 1, partz_ids[0] = npz, partz_nz[0] = nz

    int ipz        = 0;
    int ipartz_ids = 0;
    for (int i = 0; i < geom.npartz; ++i) {
        int ipart_nz = geom.partz_nz[i];
        ipartz_ids   = geom.partz_ids[i] - ipartz_ids;
        if (iz <= ipart_nz * ipartz_ids) {
            ipz += iz / ipart_nz;
            break;
        } else {
            ipz += ipartz_ids;
            iz -= ipart_nz * ipartz_ids;
        }
    }
    //  global_int_t ipz = iz/geom.nz;
    int ipy  = iy / geom.ny;
    int ipx  = ix / geom.nx;
    int rank = ipx + ipy * geom.npx + ipz * geom.npy * geom.npx;
    return rank;
}

inline std::array<int, 3> get_local_3d_from_flattened(const int i, const std::array<int, 3> n)
{
    const int iz = i / (n[2] * n[1]);
    const int iy = (i - iz * n[2] * n[1]) / n[2];
    const int ix = i - iz * n[2] * n[1] - iy * n[2];
    return std::array<int, 3>{iz, iy, ix};
}


#endif // GEOMETRY_HPP
