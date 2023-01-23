// Function declarations for cuda functions.

// CUDA runtime
#include <cuda_runtime.h>
#include "rapidjson/document.h"
// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

int simplePrintf(const rapidjson::Document &);
int sysMatCalc_1z_1z(float *, const rapidjson::Document &, int);