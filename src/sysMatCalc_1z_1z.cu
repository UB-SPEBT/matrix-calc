/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED POF THE POSSIBILITY OF SUCH DAMAGE.
 */

// System includes
#include "cuda_config_input.h"
#include "myCUDA_functions.hh"
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "rapidjson/filereadstream.h"
#include <assert.h>
#include <cmath>
#include <cstdlib>
#include <fstream>
// Cuda Thrust
#include <thrust/device_vector.h>

#include <stdio.h>
#include <vector>
using std::vector;

__global__ void attenuation_kernel(float *d_AttenuArr, float *d_BorderX,
                                   float *d_BorderY, float *d_BorderZ,
                                   float *d_coeffsMap, float *d_coordMap,
                                   float *d_indexMap, configList confs)

{
  int thread_x_grid = blockIdx.x * blockDim.x + threadIdx.x;
  int thread_y_grid = blockIdx.y * blockDim.y + threadIdx.y;
  if (thread_y_grid > confs.total_N_subImg - 1) {
    return;
  }
  if (thread_x_grid > confs.total_N_subDet * confs.total_N_subCells - 1) {
    return;
  }
  // The thread_x_grid is used as the index of the sub cell along the
  // path of the gamma-ray.
  // We can also calculate the sub detector element index from the
  // thread_x_grid
  // sub detector element index
  int subDetCudaIdx = thread_x_grid / confs.total_N_subCells;
  int subDetIdxX = d_indexMap[subDetCudaIdx * 3];
  int subDetIdxY = d_indexMap[subDetCudaIdx * 3 + 1];
  int subDetIdxZ = d_indexMap[subDetCudaIdx * 3 + 2];
  // sub cell index
  int subCellCudaIdx = thread_x_grid % confs.total_N_subCells;
  // data array index
  int arrIdx = subDetCudaIdx * confs.total_N_subImg + thread_y_grid;
  // We use thread index x as i' for the detector sub-voxel,
  // and y as j' for the image voxel. We are handling only one
  // module here, so z coordinates should be given.
  // Each thread calculate for a detector-sub-voxel-image-sub-voxel pair.
  // The summation is dealt with later.

  // detector sub voxel coordinates
  float subDetCoordX = d_coordMap[subDetCudaIdx * 3];
  float subDetCoordY = d_coordMap[subDetCudaIdx * 3 + 1];
  float subDetCoordZ = d_coordMap[subDetCudaIdx * 3 + 2];

  // Find the image (FOV) voxel indices.
  // [subImgIdZ][subImgIdY][subImgIdX]
  int subImgIdX =
      thread_y_grid %
      (confs.N_imgX * confs.N_imgSubX * confs.N_imgY * confs.N_imgSubY) %
      (confs.N_imgX * confs.N_imgSubX);
  int subImgIdY =
      thread_y_grid %
      (confs.N_imgX * confs.N_imgSubX * confs.N_imgY * confs.N_imgSubY) /
      (confs.N_imgX * confs.N_imgSubX);
  int subImgIdZ = thread_y_grid / (confs.N_imgX * confs.N_imgSubX *
                                   confs.N_imgY * confs.N_imgSubY);

  // Coordinates of the image sub voxel
  // If imgLenX = 1 mm , confs.N_imgSubX = 2,
  // then subImgCoordX = 0.25 with subImgIdx = 0,
  // and  subImgCoordX = 0.75 with subImgIdx = 1.
  // If N_imgSubX = 4,
  // then subImgCoordX = 0.125 with subImgIdx = 0,
  // and  subImgCoordX = 0.375 with subImgIdx = 1.
  // However we will move the origin of the image space
  // to the center

  float subImgCoordX =
      (0.5 + subImgIdX) * (confs.imgSubLenX) - confs.imgCenterX;
  float subImgCoordY =
      (0.5 + subImgIdY) * (confs.imgSubLenY) - confs.imgCenterY;
  float subImgCoordZ =
      (0.5 + subImgIdZ) * (confs.imgSubLenZ) - confs.imgCenterZ;

  // Rotational
  // Angle in radians
  float angleRad = confs.theta * PI / 180.0;
  subImgCoordX = subImgCoordX * cos(angleRad) + subImgCoordY * sin(angleRad);
  subImgCoordY = subImgCoordY * cos(angleRad) - subImgCoordX * sin(angleRad);
  // Translational transformation
  subImgCoordX = subImgCoordX - confs.fov_radius;
  subImgCoordY = subImgCoordY - confs.panelW;
  // components of the image-to-detector vector in
  // the detector coordinate system
  float rx = subDetCoordX - subImgCoordX;
  float ry = subDetCoordY - subImgCoordY;
  float rz = subDetCoordZ - subImgCoordZ;

  // the detector sub voxel indices.
  const int subCellIdX = subCellCudaIdx / (confs.N_subCellsY * confs.N_detSubZ);
  const int subCellIdY =
      (subCellCudaIdx % (confs.N_subCellsY * confs.N_detSubZ)) /
      confs.N_detSubZ;
  const int subCellIdZ =
      (subCellCudaIdx % (confs.N_subCellsY * confs.N_detSubZ)) %
      confs.N_detSubZ;
  const float atten_coeff =
      d_coeffsMap[subCellIdX + subCellIdY * confs.N_subCellsX];
  if (!(atten_coeff > 0)) {
    return;
  }
  // Find the intercepts
  float t0 = 2;
  float t1 = 2;
  int findNts = 0;
  float x0 = d_BorderX[subCellIdX] - subImgCoordX;
  float x1 = d_BorderX[subCellIdX + 1] - subImgCoordX;
  float y0 = d_BorderY[subCellIdY] - subImgCoordY;
  float y1 = d_BorderY[subCellIdY + 1] - subImgCoordY;
  float z0 = d_BorderZ[subCellIdZ] - subImgCoordZ;
  float z1 = d_BorderZ[subCellIdZ + 1] - subImgCoordZ;

  float t_x0 = x0 / rx;
  float t_x1 = x1 / rx;
  float t_y0 = y0 / ry;
  float t_y1 = y1 / ry;
  float t_z0 = z0 / rz;
  float t_z1 = z1 / rz;

  bool t_x0_between_y = (t_x0 * ry - y0) * (t_x0 * ry - y1) > 0 ||
                        (t_x0 * ry - y0) * (t_x0 * ry - y1) == 0;
  bool t_x0_between_z = (t_x0 * rz - z0) * (t_x0 * rz - z1) > 0 ||
                        (t_x0 * rz - z0) * (t_x0 * rz - z1) == 0;
  if (t_x0_between_y && t_x0_between_z) {
    t0 = t_x0;
    findNts++;
  }

  bool t_x1_between_y = (t_x1 * ry - y0) * (t_x1 * ry - y1) > 0 ||
                        (t_x1 * ry - y0) * (t_x1 * ry - y1) == 0;
  bool t_x1_between_z = (t_x1 * rz - z0) * (t_x1 * rz - z1) > 0 ||
                        (t_x1 * rz - z0) * (t_x1 * rz - z1) == 0;
  if (t_x1_between_y && t_x1_between_z) {
    t1 = t_x1;
    findNts++;
  }

  bool t_y0_between_x = (t_y0 * rx - x0) * (t_y0 * rx - x1) > 0 ||
                        (t_y0 * rx - x0) * (t_y0 * rx - x1) == 0;
  bool t_y0_between_z = (t_y0 * rz - z0) * (t_y0 * rz - z1) > 0 ||
                        (t_y0 * rz - z0) * (t_y0 * rz - z1) == 0;
  if (t_y0_between_x && t_y0_between_z) {
    if (t0 == 2) {
      t0 = t_y0;
      findNts++;
    }

    else if (t1 == 2) {
      t1 = t_y0;
      findNts++;
    }
  }

  bool t_y1_between_x = (t_y1 * rx - x0) * (t_y1 * rx - x1) > 0 ||
                        (t_y1 * rx - x0) * (t_y1 * rx - x1) == 0;
  bool t_y1_between_z = (t_y1 * rz - z0) * (t_y1 * rz - z1) > 0 ||
                        (t_y1 * rz - z0) * (t_y1 * rz - z1) == 0;
  if (t_y1_between_x && t_y1_between_z) {
    if (t0 == 2) {
      t0 = t_y1;
      findNts++;
    }

    else if (t1 == 2) {
      t1 = t_y1;
      findNts++;
    }
  }

  bool t_z0_between_x = (t_z0 * rx - x0) * (t_z0 * rx - x1) > 0 ||
                        (t_z0 * rx - x0) * (t_z0 * rx - x1) == 0;
  bool t_z0_between_y = (t_z0 * ry - y0) * (t_z0 * ry - y1) > 0 ||
                        (t_z0 * ry - y0) * (t_z0 * ry - y1) == 0;
  if (t_z0_between_x && t_z0_between_y) {
    if (t0 == 2) {
      t0 = t_z0;
      findNts++;
    }

    else if (t1 == 2) {
      t1 = t_z0;
      findNts++;
    }
  }

  bool t_z1_between_x = (t_z1 * rx - x0) * (t_z1 * rx - x1) > 0 ||
                        (t_z1 * rx - x0) * (t_z1 * rx - x1) == 0;
  bool t_z1_between_y = (t_z1 * ry - y0) * (t_z1 * ry - y1) > 0 ||
                        (t_z1 * ry - y0) * (t_z1 * ry - y1) == 0;
  if (t_z1_between_x && t_z1_between_y) {
    if (t0 == 2) {
      t0 = t_z1;
      findNts++;
    }

    else if (t1 == 2) {
      t1 = t_z1;
      findNts++;
    }
  }
  if (findNts != 2) {
    return;
  }
  float x_t0 = t0 * rx + subImgCoordX;
  float y_t0 = t0 * ry + subImgCoordY;
  float z_t0 = t0 * rz + subImgCoordZ;
  float x_t1 = t1 * rx + subImgCoordX;
  float y_t1 = t1 * ry + subImgCoordY;
  float z_t1 = t1 * rz + subImgCoordZ;

  // printf("N t found: %d for IMG(%d,%d,%d) and DET(%d,%d,%d) on "
  //        "CELL(%d,%d,%d)\nt0: %.3f (%.3f,%.3f,%.3f)\n"
  //        "t1: %.3f (%.3f,%.3f,%.3f)\n",
  //        findNts, subImgIdX, subImgIdY, subImgIdZ, subDetIdxX, subDetIdxY,
  //        subDetIdxZ, subCellIdX, subCellIdY, subCellIdZ, t0, x_t0, y_t0, z_t0,
  //        t1, x_t1, y_t1, z_t1);
  printf("N t found: %d for IMG(%.3f,%.3f,%.3f) and DET(%f,%f,%f) on "
         "CELL(%d,%d,%d)\nt0: %.3f (%.3f,%.3f,%.3f)\n"
         "t1: %.3f (%.3f,%.3f,%.3f)\n",
         findNts, subImgCoordX, subImgCoordY, subImgCoordZ, subDetCoordX, subDetCoordY,
         subDetCoordZ, subCellIdX, subCellIdY, subCellIdZ, t0, x_t0, y_t0, z_t0,
         t1, x_t1, y_t1, z_t1);
}
__global__ void solidAngle_kernel(float *d_solidAngleArr, float *d_coordMap,
                                  float *d_indexMap, configList confs) {
  int thread_x_grid = blockIdx.x * blockDim.x + threadIdx.x;
  int thread_y_grid = blockIdx.y * blockDim.y + threadIdx.y;
  // Data array element index.
  int arrIdx = thread_x_grid * confs.total_N_subImg + thread_y_grid;
  // We use thread index x as i' for the detector sub-voxel,
  // and y as j' for the image voxel. We are handling only one
  // module here, so z coordinates should be given.
  // Each thread calculate for a detector-sub-voxel-image-sub-voxel pair.
  // The summation is dealt with later.

  // detector sub voxel coordinates
  const int subDetCoordX = d_coordMap[thread_x_grid * 3];
  const int subDetCoordY = d_coordMap[thread_x_grid * 3 + 1];
  const int subDetCoordZ = d_coordMap[thread_x_grid * 3 + 2];
  // Find the image (FOV) voxel indices.
  // [subImgIdZ][subImgIdY][subImgIdX]
  int subImgIdX = thread_y_grid % (confs.N_imgX * confs.N_imgSubX);
  int subImgIdY = thread_y_grid % (confs.N_imgX * confs.N_imgSubX *
                                   confs.N_imgY * confs.N_imgSubY);
  int subImgIdZ = thread_y_grid / (confs.N_imgX * confs.N_imgSubX *
                                   confs.N_imgY * confs.N_imgSubY);

  // Coordinates of the image sub voxel
  // If imgLenX = 1 mm , confs.N_imgSubX = 2,
  // then subImgCoordX = 0.25 with subImgIdx = 0,
  // and  subImgCoordX = 0.75 with subImgIdx = 1.
  // If N_imgSubX = 4,
  // then subImgCoordX = 0.125 with subImgIdx = 0,
  // and  subImgCoordX = 0.375 with subImgIdx = 1.
  // However we will move the origin of the image space
  // to the center

  float subImgCoordX =
      (0.5 + subImgIdX) * (confs.imgSubLenX) - confs.imgCenterX;
  float subImgCoordY =
      (0.5 + subImgIdY) * (confs.imgSubLenY) - confs.imgCenterY;
  float subImgCoordZ =
      (0.5 + subImgIdZ) * (confs.imgSubLenZ) - confs.imgCenterZ;

  // Rotational
  // Angle in radians
  float angleRad = confs.theta * PI / 180.0;
  subImgCoordX = subImgCoordX * cos(angleRad) + subImgCoordY * sin(angleRad);
  subImgCoordY = subImgCoordY * cos(angleRad) - subImgCoordX * sin(angleRad);
  // Translational transformation
  subImgCoordX = subImgCoordX - confs.fov_radius;
  subImgCoordY = subImgCoordY - confs.panelW;

  // components of the image-to-detector vector in
  // the detector coordinate system
  float rx = subDetCoordX - subImgCoordX;
  float ry = subDetCoordY - subImgCoordY;
  float rz = subDetCoordZ - subImgCoordZ;
  float totalDist = sqrt(rx * rx + ry * ry + rz * rz);

  // Solid angle calculation
  // The total solid angle of the cuboid to the image voxel may has
  // contribution from five faces (never the face in the back). Normal vectors
  // are pointing to the center of the cuboid. In the detector coordinate
  // system normal vectors are: (1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)

  float subDetLenX = confs.XtalLengthX / confs.N_detSubX;
  float subDetLenY = confs.XtalLengthY / confs.N_detSubY;
  float subDetLenZ = confs.XtalLengthZ / confs.N_XtalSegZ / confs.N_detSubZ;
  float subDetAreaX = subDetLenY * subDetLenZ;
  float subDetAreaY = subDetLenX * subDetLenZ;
  float subDetAreaZ = subDetLenX * subDetLenY;
  float areaDotProductSum = 0;
  if (rx >= 0)
    areaDotProductSum += rx * subDetAreaX;
  if (ry >= 0)
    areaDotProductSum += ry * subDetAreaY;
  else
    areaDotProductSum += -ry * subDetAreaY;
  if (rz >= 0)
    areaDotProductSum += rz * subDetAreaZ;
  else
    areaDotProductSum += -rz * subDetAreaZ;

  d_solidAngleArr[arrIdx] =
      areaDotProductSum / (totalDist * totalDist * totalDist);
}

int calDetCoordinates(std::vector<std::vector<float>> &coordMap,
                      std::vector<std::vector<int>> &indexMap,
                      std::vector<float> &vecBorderX,
                      std::vector<float> &vecBorderY,
                      std::vector<float> &vecBorderZ,
                      std::vector<std::vector<float>> &vec_h_coeffsMap,
                      int zIdx_det, const rapidjson::Document &jsonDoc) {

  printf("Number of x borders: %zu\n", vecBorderX.size());
  printf("Number of y borders: %zu\n", vecBorderY.size());
  printf("Number of z borders: %zu\n", vecBorderZ.size());
  printf("Shape of the coeffs map: %zu x %zu\n", vec_h_coeffsMap.size(),
         vec_h_coeffsMap[0].size());

  const rapidjson::Value &jsonCoeffs =
      jsonDoc["Detector"]["Attenuation Constants"];
  assert(jsonCoeffs.IsArray());
  float xtalCoeff = jsonCoeffs.GetArray()[1].GetFloat();

  int N_detSubZ = jsonDoc["Detector"]["Detector N Subdivision Z"].GetInt();
  // int N_moduleZ = jsonDoc["Detector"]["N Modules Axial"].GetInt();
  // int N_XtalSegZ = jsonDoc["Detector"]["N Crystal Segments"].GetInt();
  for (int ix = 0; ix < vec_h_coeffsMap.size(); ix++) {
    for (int iy = 0; iy < vec_h_coeffsMap[0].size(); iy++) {

      float coordX = (vecBorderX[ix] + vecBorderX[ix + 1]) * 0.5;
      float coordY = (vecBorderY[iy] + vecBorderY[iy + 1]) * 0.5;
      for (int iz = 0; iz < N_detSubZ; iz++) {
        float coordZ = (vecBorderZ[iz + zIdx_det * N_detSubZ] +
                        vecBorderZ[iz + zIdx_det * N_detSubZ + 1]) *
                       0.5;
        // vector<float> thisCoords{coordX,coordY,coordZ};
        if (vec_h_coeffsMap[ix][iy] == xtalCoeff) {
          coordMap.push_back(vector<float>{coordX, coordY, coordZ});
          indexMap.push_back(vector<int>{ix, iy, iz + zIdx_det * N_detSubZ});
        }

        // thisCoords.push_back(coordX);
        // thisCoords.push_back(coordY);
        // thisCoords.push_back(coordZ);
      }
    }
  }
  // printf("CoordMap Dimension (X x Y x Z x 3): %zu x %zu x %zu x %zu\n",
  //        coordMap.size(), coordMap[0].size(), coordMap[0][0].size(),
  //        coordMap[0][0][0].size());

  printf("Crystal Subdivision Coordinates Map Dimension (N x 3): %zu x %zu\n",
         coordMap.size(), coordMap[0].size());
  return EXIT_SUCCESS;
}

int sysMatCalc_1z_1z(float *dataArr, std::vector<float> &vecBorderX,
                     std::vector<float> &vecBorderY,
                     std::vector<float> &vecBorderZ,
                     std::vector<std::vector<float>> &vec_h_coeffsMap,
                     const rapidjson::Document &jsonDoc, int NElement) {
  int cuda_devID = jsonDoc["CUDA setting"]["CUDA Device ID"].GetInt();
  int N_cudaDev;
  checkCudaErrors(cudaGetDeviceCount(&N_cudaDev));
  if (cuda_devID > N_cudaDev - 1) {
    fprintf(stderr, "Using CUDA Device ID: %d\n", cuda_devID);
    return EXIT_FAILURE;
  }

  cudaDeviceProp props;

  // Get GPU information

  checkCudaErrors(cudaSetDevice(cuda_devID));
  checkCudaErrors(cudaGetDeviceProperties(&props, cuda_devID));
  printf("Device %d: \"%s\" with Compute %d.%d capability\n", cuda_devID,
         props.name, props.major, props.minor);
  float *d_dataArr;
  float *d_BorderX, *d_BorderY, *d_BorderZ, *d_coeffsMap;
  int coeffsMap_size =
      vec_h_coeffsMap.size() * vec_h_coeffsMap[0].size() * sizeof(float);
  float *h_coeffsMap = (float *)malloc(coeffsMap_size);

  std::vector<std::vector<float>> vec_coordMap;
  std::vector<std::vector<int>> vec_indexMap;
  int zIdx_det = 0;
  calDetCoordinates(vec_coordMap, vec_indexMap, vecBorderX, vecBorderY,
                    vecBorderZ, vec_h_coeffsMap, zIdx_det, jsonDoc);

  struct configList confs;
  vector<int> N_borders(3, 0);
  N_borders[0] = vecBorderX.size();
  N_borders[1] = vecBorderY.size();
  N_borders[2] = vecBorderZ.size();
  printf("CoordMap Size: %zu\n", vec_coordMap.size());
  initConfList(&confs, jsonDoc, 0, N_borders, vec_coordMap.size());

  // for (auto v : indexMap) {
  //   printf("(%d,%d,%d)", v[0], v[1], v[2]);
  // }
  // std::cout << std::endl;
  float *h_coordMap = (float *)malloc(vec_coordMap.size() * 3 * sizeof(float));
  float *h_indexMap = (float *)malloc(vec_indexMap.size() * 3 * sizeof(int));

  if (h_coeffsMap == NULL || h_coordMap == NULL || h_indexMap == NULL) {
    fprintf(stderr, "Failed to allocate host vectors!\n");
    exit(EXIT_FAILURE);
  }
  for (int idx = 0; idx < vec_coordMap.size(); idx++) {
    h_coordMap[idx * 3] = vec_coordMap[idx][0];
    h_coordMap[idx * 3 + 1] = vec_coordMap[idx][1];
    h_coordMap[idx * 3 + 2] = vec_coordMap[idx][2];
    h_indexMap[idx * 3] = vec_indexMap[idx][0];
    h_indexMap[idx * 3 + 1] = vec_indexMap[idx][1];
    h_indexMap[idx * 3 + 2] = vec_indexMap[idx][2];
  }
  for (int ix = 0; ix < vec_h_coeffsMap.size(); ix++) {
    for (int iy = 0; iy < vec_h_coeffsMap[0].size(); iy++) {
      h_coeffsMap[ix + iy * vec_h_coeffsMap.size()] = vec_h_coeffsMap[ix][iy];
    }
  }
  float *d_coordMap, *d_indexMap;
  checkCudaErrors(cudaMalloc((void **)&d_coordMap,
                             vec_coordMap.size() * 3 * sizeof(float)));
  checkCudaErrors(
      cudaMalloc((void **)&d_indexMap, vec_indexMap.size() * 3 * sizeof(int)));
  checkCudaErrors(cudaMalloc((void **)&d_coeffsMap, coeffsMap_size));

  int arrSize = NElement * sizeof(float);
  // printf("N Total Matrix Elements:\t%d\nArray Size:\t%d\n", NElement,
  // arrSize);
  checkCudaErrors(cudaMalloc((void **)&d_dataArr, arrSize));
  checkCudaErrors(
      cudaMalloc((void **)&d_BorderX, vecBorderX.size() * sizeof(float)));
  checkCudaErrors(
      cudaMalloc((void **)&d_BorderY, vecBorderY.size() * sizeof(float)));
  checkCudaErrors(
      cudaMalloc((void **)&d_BorderZ, vecBorderZ.size() * sizeof(float)));

  float *h_BorderX = (float *)malloc(vecBorderX.size() * sizeof(float));
  float *h_BorderY = (float *)malloc(vecBorderY.size() * sizeof(float));
  float *h_BorderZ = (float *)malloc(vecBorderZ.size() * sizeof(float));
  // Verify that allocations succeeded
  if (h_BorderX == NULL || h_BorderY == NULL || h_BorderZ == NULL) {
    fprintf(stderr, "Failed to allocate host vectors!\n");
    exit(EXIT_FAILURE);
  }

  for (int idx = 0; idx < vecBorderX.size(); idx++) {
    h_BorderX[idx] = vecBorderX[idx];
  }
  for (int idx = 0; idx < vecBorderY.size(); idx++) {
    h_BorderY[idx] = vecBorderY[idx];
  }
  for (int idx = 0; idx < vecBorderZ.size(); idx++) {
    h_BorderZ[idx] = vecBorderZ[idx];
  }

  checkCudaErrors(cudaMemcpy(d_BorderX, h_BorderX,
                             vecBorderX.size() * sizeof(float),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_BorderY, h_BorderY,
                             vecBorderY.size() * sizeof(float),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_BorderZ, h_BorderZ,
                             vecBorderZ.size() * sizeof(float),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_coordMap, h_coordMap,
                             vec_coordMap.size() * 3 * sizeof(float),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_indexMap, h_indexMap,
                             vec_indexMap.size() * 3 * sizeof(int),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_coeffsMap, h_coeffsMap, coeffsMap_size,
                             cudaMemcpyHostToDevice));

  // Kernel configuration, where a 2d grid and
  // 2d blocks are configured.

  int dimBlockX = jsonDoc["CUDA setting"]["Block Dim X"].GetInt();
  int dimBlockY = jsonDoc["CUDA setting"]["Block Dim Y"].GetInt();
  // int dimGridX = N_blockX * N_blockY * N_cellX * N_cellY / dimBlockX;
  // int dimGridX = vec_coordMap.size() / dimBlockX;
  // int dimGridY = confs.N_imgX * confs.N_imgY * confs.N_imgSubX *
  //                confs.N_imgSubY * confs.N_imgSubZ / dimBlockY;
  // printf("CUDA Kernel Execution Configuration:\n");
  // printf("  Grid Dimension:  %4d x %d %8s\n", dimGridX, dimGridY,
  // "blocks."); printf("  Block Dimension: %4d x %d %8s\n", dimBlockX,
  // dimBlockY, "threads.");
  dim3 dimGrid;
  dim3 dimBlock(dimBlockX, dimBlockY);
  // float t[subDetIdX + subDetIdY + subDetIdZ + 3];
  int subDivArrSize = vec_coordMap.size() * confs.N_imgX * confs.N_imgY *
                      confs.N_imgSubX * confs.N_imgSubY * confs.N_imgSubZ;

  float *d_solidAngleArr, *d_attenTermArr;
  checkCudaErrors(
      cudaMalloc((void **)&d_solidAngleArr, subDivArrSize * sizeof(float)));
  checkCudaErrors(
      cudaMalloc((void **)&d_attenTermArr, subDivArrSize * sizeof(float)));
  dimGrid.x = ceil((float)vec_coordMap.size() / dimBlockX);
  dimGrid.y = ceil(float(confs.N_imgX * confs.N_imgY * confs.N_imgSubX *
                         confs.N_imgSubY * confs.N_imgSubZ) /
                   dimBlockY);
  printf("  Block Dimension: %4d x %d %8s\n", dimBlockX, dimBlockY, "threads.");
  printf("  Grid Dimension:  %4d x %d %8s\n", dimGrid.x, dimGrid.y, "blocks.");
  solidAngle_kernel<<<dimGrid, dimBlock>>>(d_solidAngleArr, d_coordMap,
                                           d_indexMap, confs);
  dimGrid.x =
      ceil((float)(vec_coordMap.size() * confs.total_N_subCells) / dimBlockX);
  printf("Total x elements: %f\n",
         (float)(vec_coordMap.size() * confs.total_N_subCells));
  printf("  Grid Dimension:  %4d x %d %8s\n", dimGrid.x, dimGrid.y, "blocks.");
  attenuation_kernel<<<dimGrid, dimBlock>>>(d_attenTermArr, d_BorderX,
                                            d_BorderY, d_BorderZ, d_coeffsMap,
                                            d_coordMap, d_indexMap, confs);
  // sysMatCalc_1z_1z_kernel<<<dimGrid, dimBlock>>>(
  //     d_dataArr, d_BorderX, d_BorderY, d_BorderZ, d_coeffsMap,
  //     d_coordMap, d_indexMap, d_intercept_t, confs);
  cudaDeviceSynchronize();

  // check for error
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  // checkCudaErrors(cudaMemcpy(tempArr, d_dataArr, arrSize,
  // cudaMemcpyDeviceToHost)); dataArr = tempArr;
  checkCudaErrors(
      cudaMemcpy(dataArr, d_dataArr, arrSize, cudaMemcpyDeviceToHost));
  // free(tempArr);
  cudaFree(d_dataArr);
  cudaFree(d_BorderX);
  cudaFree(d_BorderY);
  cudaFree(d_BorderZ);
  cudaFree(d_coeffsMap);

  free(h_BorderX);
  free(h_BorderY);
  free(h_BorderZ);

  return EXIT_SUCCESS;
}