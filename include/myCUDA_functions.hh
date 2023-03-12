// Function declarations for cuda functions.

// CUDA runtime
#include "rapidjson/document.h"
#include <cuda_runtime.h>
// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

int simplePrintf(const rapidjson::Document &);
int sysMatCalc_1z_1z(float *, std::vector<float> &, std::vector<float> &,
                     std::vector<float> &, std::vector<std::vector<float>> &,
                     const rapidjson::Document &, int);