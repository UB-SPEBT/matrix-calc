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

__global__ void sysMatCalc_1z_1z_kernel(float *d_dataArr, configList confs)
// __global__ void testKernel()
{
  int thread_x_grid = blockIdx.x * blockDim.x + threadIdx.x;
  int thread_y_grid = blockIdx.y * blockDim.y + threadIdx.y;
  int threadIdx_grid = thread_y_grid * blockDim.x * gridDim.x + thread_x_grid;

  // printf("Global Thread# %d\t[x: %d, y: %d], val: %f\n", threadIdx_grid, thread_x_grid, thread_y_grid, confs.cellLenX);
  d_dataArr[threadIdx_grid] = (float)threadIdx_grid;
}

int sysMatCalc_1z_1z(float *dataArr, const rapidjson::Document &jsonDoc, int NElement)
{
  int devID = jsonDoc["CUDA setting"].FindMember("CUDA Device ID")->value.GetInt();
  int dimGridx = jsonDoc["CUDA setting"]["Grid Dim X"].GetInt();
  int dimGridy = jsonDoc["CUDA setting"]["Grid Dim Y"].GetInt();
  int dimBlockx = jsonDoc["CUDA setting"]["Block Dim X"].GetInt();
  int dimBlocky = jsonDoc["CUDA setting"]["Block Dim Y"].GetInt();
  cudaDeviceProp props;

  struct configList confs;
  initConfList(&confs, jsonDoc);

  // Get GPU information
  checkCudaErrors(cudaGetDevice(&devID));
  checkCudaErrors(cudaGetDeviceProperties(&props, devID));
  printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name,
         props.major, props.minor);

  float *d_dataArr;
  // dataArr = (float *)malloc(NElement);
  int arrSize = NElement * sizeof(float);
  // printf("N Total Matrix Elements:\t%d\nArray Size:\t%d\n", NElement, arrSize);
  cudaMalloc((void **)&d_dataArr, arrSize);

  // Kernel configuration, where a 2d grid and
  // 2d blocks are configured.
  dim3 dimGrid(dimGridx, dimGridy);
  dim3 dimBlock(dimBlockx, dimBlocky);

  sysMatCalc_1z_1z_kernel<<<dimGrid, dimBlock>>>(d_dataArr, confs);
  cudaDeviceSynchronize();

  // check for error
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  // checkCudaErrors(cudaMemcpy(tempArr, d_dataArr, arrSize, cudaMemcpyDeviceToHost));
  // dataArr = tempArr;
  checkCudaErrors(cudaMemcpy(dataArr, d_dataArr, arrSize, cudaMemcpyDeviceToHost));
  // free(tempArr);
  cudaFree(d_dataArr);
  return EXIT_SUCCESS;
}