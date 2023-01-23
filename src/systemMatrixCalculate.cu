// System includes
#include <stdio.h>
#include <assert.h>

#include "myCUDA_functions.hh"
#include "kernel_config.h"

__global__ void sysMatCalc_1z_1z_kernel(int val)
{
  int blockIdx_grid = blockIdx.y * gridDim.x + blockIdx.x;
  int threadIdx_block = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
  int threadIdx_grid = threadIdx_block + blockIdx_grid * blockDim.x * blockDim.y * blockDim.z;
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  printf("[block# %d, thread# %d]:\t\tGlobal Thread# %d\t[row: %d, col: %d]\n", blockIdx_grid,
         threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
             threadIdx.x,
         threadIdx_grid, row, col);
}

int sysMatCalc_1z_1z(const rapidjson::Document &jsonDoc)
{
  int devID = jsonDoc["CUDA setting"].FindMember("CUDA device ID")->value.GetInt();
  int dimGridx = jsonDoc["CUDA setting"].FindMember("grid dim x")->value.GetInt();
  int dimGridy = jsonDoc["CUDA setting"].FindMember("grid dim y")->value.GetInt();
  int dimBlockx = jsonDoc["CUDA setting"].FindMember("block dim x")->value.GetInt();
  int dimBlocky = jsonDoc["CUDA setting"].FindMember("block dim y")->value.GetInt();
  cudaDeviceProp props;

  // Get GPU information
  checkCudaErrors(cudaGetDevice(&devID));
  checkCudaErrors(cudaGetDeviceProperties(&props, devID));
  printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name,
         props.major, props.minor);

  // Kernel configuration, where a 2d grid and
  // 2d blocks are configured.
  dim3 dimGrid(dimGridx, dimGridy);
  dim3 dimBlock(dimBlockx, dimBlocky);
  sysMatCalc_1z_1z_kernel<<<dimGrid, dimBlock>>>(10);
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