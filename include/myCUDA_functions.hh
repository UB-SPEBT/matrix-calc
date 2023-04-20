// Function declarations for cuda functions.

// CUDA runtime
#include "rapidjson/document.h"
#include <boost/program_options/variables_map.hpp>
#include <cuda_runtime.h>
// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/support/date_time.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/utility/setup/file.hpp>
#include "cmd_args.hh"
int simplePrintf(const rapidjson::Document &);
int sysMatCalc_1z_1z(
    float *, std::vector<float> &, std::vector<float> &, std::vector<float> &,
    std::vector<std::vector<float>> &, const rapidjson::Document &, int, po::variables_map &);