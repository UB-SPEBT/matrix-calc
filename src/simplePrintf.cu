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
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// System includes
#include <stdio.h>
#include <assert.h>
#include "cuda_config_input.h"
#include "myCUDA_functions.hh"
#include "rapidjson/document.h"

__global__ void testKernel(configList confs)
// __global__ void testKernel()
{
  int blockIdx_grid = blockIdx.y * gridDim.x + blockIdx.x;
  int threadIdx_block = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
  int threadIdx_grid = threadIdx_block + blockIdx_grid * blockDim.x * blockDim.y * blockDim.z;
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  printf("[block# %d, thread# %d]:\t\tGlobal Thread# %d\t[row: %d, col: %d], val: %f\n", blockIdx_grid,
         threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
             threadIdx.x,
         threadIdx_grid, row, col, confs.detectorLenX);
}

__global__ void sysMatCalc_1z_1z_kernel(configList confs)
// __global__ void testKernel()
{
  int threadIdx_grid = threadIdx.y * blockDim.x + threadIdx.x + (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y;
  int thread_x_grid = blockIdx.x * blockDim.x + threadIdx.x;
  int thread_y_grid = blockIdx.y * blockDim.y + threadIdx.y;
  printf("Global Thread# %d\t[x: %d, y: %d], val: %f\n",
         threadIdx_grid, thread_x_grid, thread_y_grid, confs.detectorLenX);
}

int simplePrintf(const rapidjson::Document &jsonDoc)
{
  int devID = jsonDoc["CUDA setting"].FindMember("CUDA device ID")->value.GetInt();
  int dimGridx = jsonDoc["CUDA setting"].FindMember("grid dim x")->value.GetInt();
  int dimGridy = jsonDoc["CUDA setting"].FindMember("grid dim y")->value.GetInt();
  int dimBlockx = jsonDoc["CUDA setting"].FindMember("block dim x")->value.GetInt();
  int dimBlocky = jsonDoc["CUDA setting"].FindMember("block dim y")->value.GetInt();
  cudaDeviceProp props;

  struct configList confs;
  initConfList(&confs, jsonDoc);

  // Get GPU information
  checkCudaErrors(cudaGetDevice(&devID));
  checkCudaErrors(cudaGetDeviceProperties(&props, devID));
  printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name,
         props.major, props.minor);

  // Kernel configuration, where a 2d grid and
  // 2d blocks are configured.
  dim3 dimGrid(dimGridx, dimGridy);
  dim3 dimBlock(dimBlockx, dimBlocky);

  testKernel<<<dimGrid, dimBlock>>>(confs);

  cudaDeviceSynchronize();
  // check for error
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }

  return EXIT_SUCCESS;
}

int sysMatCalc_1z_1z(const rapidjson::Document &)
{
  int devID = jsonDoc["CUDA setting"].FindMember("CUDA device ID")->value.GetInt();
  int dimGridx = jsonDoc["CUDA setting"].FindMember("grid dim x")->value.GetInt();
  int dimGridy = jsonDoc["CUDA setting"].FindMember("grid dim y")->value.GetInt();
  int dimBlockx = jsonDoc["CUDA setting"].FindMember("block dim x")->value.GetInt();
  int dimBlocky = jsonDoc["CUDA setting"].FindMember("block dim y")->value.GetInt();
  cudaDeviceProp props;

  struct configList confs;
  initConfList(&confs, jsonDoc);

  // Get GPU information
  checkCudaErrors(cudaGetDevice(&devID));
  checkCudaErrors(cudaGetDeviceProperties(&props, devID));
  printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name,
         props.major, props.minor);

  // Kernel configuration, where a 2d grid and
  // 2d blocks are configured.
  dim3 dimGrid(dimGridx, dimGridy);
  dim3 dimBlock(dimBlockx, dimBlocky);

  sysMatCalc_1z_1z_kernel<<<dimGrid, dimBlock>>>(confs);

  cudaDeviceSynchronize();
  // check for error
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }

  return EXIT_SUCCESS;
}