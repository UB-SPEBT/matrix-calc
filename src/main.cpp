// C/C++ standard headers
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/support/date_time.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
// Non-standard headers
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "rapidjson/error/error.h"
// Self-defined headers
#include "cmd_args.hh"
#include "config_parser.hh"
#include "myCUDA_functions.hh"

void initBoostLogger(const rapidjson::Document &jsonDoc) {
  namespace logging = boost::log;
  namespace expr = boost::log::expressions;
  namespace keywords = boost::log::keywords;
  if (jsonDoc.HasMember("Logger")) {
    if (jsonDoc["Logger"].HasMember("Level")) {
      if (jsonDoc["Logger"]["Level"].IsString()) {
        std::string logFname = "debug.log";
        if (jsonDoc["Logger"].HasMember("Filename")) {
          logFname.assign(jsonDoc["Logger"]["Filename"].GetString());
        }
        logging::add_file_log(
            keywords::file_name = logFname,
            // This makes the sink to write log records that look like this:
            // YYYY-MM-DD HH:MI:SS: <normal> A normal severity message
            // YYYY-MM-DD HH:MI:SS: <error> An error severity message
            keywords::format =
                (expr::stream
                 << expr::format_date_time<boost::posix_time::ptime>(
                        "TimeStamp", "%Y-%m-%d %H:%M:%S")
                 << ": <" << logging::trivial::severity << "> "
                 << expr::smessage));
        std::string logLevel(jsonDoc["Logger"]["Level"].GetString());
        if (logLevel.compare("INFO")) {

          boost::log::core::get()->set_filter(boost::log::trivial::severity >=
                                              boost::log::trivial::info);
        } else if (logLevel.compare("DEBUG")) {

          boost::log::core::get()->set_filter(boost::log::trivial::severity >=
                                              boost::log::trivial::debug);
        } else if (logLevel.compare("WARNING")) {

          boost::log::core::get()->set_filter(boost::log::trivial::severity >=
                                              boost::log::trivial::warning);
        } else if (logLevel.compare("TRACE")) {

          boost::log::core::get()->set_filter(boost::log::trivial::severity >=
                                              boost::log::trivial::trace);
        } else if (logLevel.compare("ERROR")) {

          boost::log::core::get()->set_filter(boost::log::trivial::severity >=
                                              boost::log::trivial::error);
        } else {
          std::cerr << "Unknown Logger Level: " << logLevel << std::endl;
          boost::log::core::get()->set_filter(boost::log::trivial::severity >=
                                              boost::log::trivial::fatal);
        }
      } else {
        boost::log::core::get()->set_filter(boost::log::trivial::severity >=
                                            boost::log::trivial::fatal);
      }
    }
  }
}
double NElem(const rapidjson::Document &jsonDoc) {
  // int N_panel = jsonDoc["Detector"]["N Panels"].GetInt();
  // int N_moduleZ = jsonDoc["Detector"]["N Modules Axial"].GetInt();
  // int N_xtalSegZ = jsonDoc["Detector"]["N Crystal Segments"].GetInt();
  int N_blockX = jsonDoc["Detector"]["Module"]["N Block X"].GetInt();
  int N_blockY = jsonDoc["Detector"]["Module"]["N Block Y"].GetInt();
  int N_cellX = jsonDoc["Detector"]["Module"]["Block"]["N Cells X"].GetInt();
  int N_cellY = jsonDoc["Detector"]["Module"]["Block"]["N Cells Y"].GetInt();
  int N_imgX = jsonDoc["Image Volume"]["N Image Voxel X"].GetInt();
  int N_imgY = jsonDoc["Image Volume"]["N Image Voxel Y"].GetInt();
  // int N_imgZ = jsonDoc["Image Volume"]["N Image Voxel Z"].GetInt();
  //   int N_imgVoxelZ = jsonDoc["Image Volume"]["N Image Voxel
  //   Z"].GetInt(); int N_imgSubX = jsonDoc["Image Volume"]["Voxel N
  //   Subdivision X"].GetInt(); int N_imgSubY = jsonDoc["Image
  //   Volume"]["Voxel N Subdivision Y"].GetInt(); int N_imgSubZ =
  //   jsonDoc["Image Volume"]["Voxel N Subdivision Z"].GetInt();

  //   return N_panel * N_moduleZ * N_blockX * N_blockY * N_cellX * N_cellY
  //   *
  //          N_xtalSegZ * N_imgVoxelX * N_imgVoxelY * N_imgVoxelZ;
  printf("  %-32s| %d x %d\n",
         "Detector space dimension (xy):", N_blockX * N_cellX,
         N_blockY * N_cellY);
  printf("  %-32s| %d x %d\n", "Image space dimension (xy):", N_imgX, N_imgY);
  printf("  %-32s| %d x %d = %d\n",
         "System matrix element size:", N_blockX * N_blockY * N_cellX * N_cellY,
         N_imgX * N_imgY,
         N_blockX * N_blockY * N_cellX * N_cellY * N_imgX * N_imgY);
  return N_blockX * N_blockY * N_cellX * N_cellY * N_imgX * N_imgY;
}
void calcBorderAndCoeff(
    std::vector<float> &borderX, std::vector<float> &borderY,
    std::vector<float> &borderZ, std::vector<std::vector<float>> &h_coeffsMap,
    const rapidjson::Document &jsonDoc,
    boost::log::sources::severity_logger<boost::log::trivial::severity_level>
        lg) {
  using namespace boost::log::trivial;
  int N_detSubX = jsonDoc["Detector"]["Detector N Subdivision X"].GetInt();
  int N_detSubY = jsonDoc["Detector"]["Detector N Subdivision Y"].GetInt();
  int N_detSubZ = jsonDoc["Detector"]["Detector N Subdivision Z"].GetInt();
  int N_moduleZ = jsonDoc["Detector"]["N Modules Axial"].GetInt();
  int N_XtalSegZ = jsonDoc["Detector"]["N Crystal Segments"].GetInt();
  float cellLengthX = jsonDoc["Detector"]["Cell Length X"].GetFloat();
  float cellLengthY = jsonDoc["Detector"]["Cell Length Y"].GetFloat();
  float XtalLengthX = jsonDoc["Detector"]["Crystal Length X"].GetFloat();
  float XtalLengthY = jsonDoc["Detector"]["Crystal Length Y"].GetFloat();
  float XtalLengthZ = jsonDoc["Detector"]["Crystal Length Z"].GetFloat();

  float panelCenterZ = N_moduleZ * XtalLengthZ * 0.5;
  // Module configuration
  int N_blockX = jsonDoc["Detector"]["Module"]["N Block X"].GetInt();
  int N_blockY = jsonDoc["Detector"]["Module"]["N Block Y"].GetInt();
  int N_cellX = jsonDoc["Detector"]["Module"]["Block"]["N Cells X"].GetInt();
  int N_cellY = jsonDoc["Detector"]["Module"]["Block"]["N Cells Y"].GetInt();
  int pattern[N_cellX][N_cellY];
  const rapidjson::Value &jsonPattern =
      jsonDoc["Detector"]["Module"]["Block"]["Pattern"];
  assert(jsonPattern.IsArray());
  auto patternArrObj = jsonPattern.GetArray();
  for (int cellIdx_y = 0; cellIdx_y < N_cellY; cellIdx_y++) {
    auto innerArr = patternArrObj[cellIdx_y].GetArray();
    for (int cellIdx_x = 0; cellIdx_x < N_cellX; cellIdx_x++) {
      pattern[cellIdx_x][cellIdx_y] = innerArr[cellIdx_x].GetInt();
      // printf("Pattern[%d][%d] = %d\n", cellIdx_x, cellIdx_y,
      // innerArr[cellIdx_x].GetInt());
    }
  }
  char message[128];
  sprintf(message, "Block Crystal Configuration Pattern (%d x %d): ", N_cellY,
          N_cellX);
  BOOST_LOG_SEV(lg, debug) << message;

  float coeffs[4];
  const rapidjson::Value &jsonCoeffs =
      jsonDoc["Detector"]["Attenuation Constants"];
  assert(jsonCoeffs.IsArray());
  auto coeffsArrObj = jsonCoeffs.GetArray();
  for (int idx = 0; idx < 4; idx++) {
    coeffs[idx] = coeffsArrObj[idx].GetFloat();
  }
  std::stringstream messagess;
  messagess << "\n";
  for (int i = 0; i < N_cellX; i++) {
    messagess << "\t";
    for (int j = 0; j < N_cellY; j++) {
      messagess << std::setw(2) << pattern[i][j];
    }
    messagess << "\n";
  }
  BOOST_LOG_SEV(lg, debug) << messagess.str();
  int N_subCellsX = N_cellX * N_blockX * (N_detSubX + 2);
  int N_subCellsY = N_cellY * N_blockY * (N_detSubY + 2);
  int N_subCellsZ = N_moduleZ * N_XtalSegZ * N_detSubZ;
  borderX.resize(N_subCellsX + 1);
  borderY.resize(N_subCellsY + 1);
  borderZ.resize(N_subCellsZ + 1);
  borderX[0] = 0;
  borderY[0] = 0;
  borderZ[0] = 0.0 - panelCenterZ;
  BOOST_LOG_SEV(lg, debug) << "Panel Center Z: " << std::setprecision(3)
                           << panelCenterZ << std::endl;
  for (int ix = 1; ix < N_subCellsX + 1; ix++) {
    int ifGap =
        floor(2.0 * abs((1.0 + 0.5 * N_detSubX) - (ix % (N_detSubX + 2) - 1)) /
              (N_detSubX + 2));
    float increment =
        ifGap ? 0.5 * (cellLengthX - XtalLengthX) : XtalLengthX / N_detSubX;
    borderX[ix] = borderX[ix - 1] + increment;
    // std::cout << borderX[ix - 1] << " ";
  }
  //   std::cout << borderX[N_subCellsX] << std::endl;
  for (int iy = 1; iy < N_subCellsY + 1; iy++) {
    int ifGap =
        floor(2.0 * abs((1.0 + 0.5 * N_detSubY) - (iy % (N_detSubY + 2) - 1)) /
              (N_detSubY + 2));
    float increment =
        ifGap ? 0.5 * (cellLengthY - XtalLengthY) : XtalLengthY / N_detSubY;
    borderY[iy] = borderY[iy - 1] + increment;
    // std::cout << borderY[iy - 1] << " ";
  }
  //   std::cout << borderY[N_subCellsY] << std::endl;
  for (int iz = 1; iz < N_subCellsZ + 1; iz++) {
    float increment = XtalLengthZ / N_XtalSegZ / N_detSubZ;
    borderZ[iz] = borderZ[iz - 1] + increment;
    // std::cout << borderZ[iz - 1] << " ";
  }
  //   std::cout << borderZ[N_subCellsZ] << std::endl;
  // printf("Pattern Size: %lu", sizeof(pattern) / sizeof(pattern[0]));

  h_coeffsMap.resize(N_subCellsX, std::vector<float>(N_subCellsY));
  for (int iy = 0; iy < N_subCellsY; iy++) {
    for (int ix = 0; ix < N_subCellsX; ix++) {
      int idxSubCellX = ix % (N_cellX * (N_detSubX + 2));
      int idxCellX = idxSubCellX / (N_detSubX + 2);
      int idxSubCellY = iy % (N_cellY * (N_detSubY + 2));
      int idxCellY = idxSubCellY / (N_detSubY + 2);
      // Sub-element indices in the cell,
      // idxSubX runs from 0 to N_detSubX + 1.
      // idxSubY runs from 0 to N_detSubY + 1.
      int idxSubX = idxSubCellX % (N_detSubX + 2);
      int idxSubY = idxSubCellY % (N_detSubY + 2);
      if (idxSubX * idxSubY == 0 or idxSubX == N_detSubX + 1 or
          idxSubY == N_detSubX + 1) {
        h_coeffsMap[ix][iy] = 0;
      } else {
        h_coeffsMap[ix][iy] = coeffs[pattern[idxCellX][idxCellY]];
      }
      //   printf("%.3f ", coeffs[pattern[idxCellX][idxCellY]]);
    }
    // std::cout << std::endl;
  }
}

int main(int argc, char **argv) {
  auto start1 = std::chrono::high_resolution_clock::now();
  std::string configPath;
  // rapidJSON document object
  rapidjson::Document RJDocument;
  using namespace boost::log;
  // parse the commandline arguments
  po::variables_map vm;
  int result = cmd_args_parser(argc, argv, vm, configPath);
  if (result)
    return 1;
  std::cout << "Using config file:\t" << configPath << '\n';
  // parse the JSON configuration file
  result = parse_config_json(configPath, RJDocument);
  if (result)
    return 1;

  // Init the logger with filter setting
  initBoostLogger(RJDocument);
  boost::log::add_common_attributes();
  boost::log::sources::severity_logger<boost::log::trivial::severity_level> lg;
  int NElement = NElem(RJDocument);
  int arrSize = NElement * sizeof(float);
  float *dataArray = (float *)malloc(arrSize);
  // log message string stream
  std::stringstream lgMsgSs;
  lgMsgSs << "Matrix N Elements:\t" << NElement << "\nMatrix Data Size:\t"
          << arrSize << "(Bytes)\n";
  BOOST_LOG_SEV(lg, trivial::debug) << lgMsgSs.str();
  std::vector<float> vecBorderX, vecBorderY, vecBorderZ;
  std::vector<std::vector<float>> h_coeffsMap;
  calcBorderAndCoeff(vecBorderX, vecBorderY, vecBorderZ, h_coeffsMap,
                     RJDocument, lg);

  //   for (auto v : h_coeffsMap) {
  //     for (auto elem : v) {
  //       std::cout << std::setw(6) << elem << " ";
  //     }
  //     std::cout << std::endl;
  //   }
  auto stop1 = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1);
  std::cout << "Initialzation complete in: " << duration.count()
            << " microseconds.\n";
  auto start2 = std::chrono::high_resolution_clock::now();
  sysMatCalc_1z_1z(dataArray, vecBorderX, vecBorderY, vecBorderZ, h_coeffsMap,
                   RJDocument, NElement, vm);
  auto stop2 = std::chrono::high_resolution_clock::now();
  duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start2);
  std::cout << "Matrix calculation (CUDA) complete in: "
            << (float)duration.count() / 1000000.0 << " seconds.\n";
  auto start3 = std::chrono::high_resolution_clock::now();
  int zIdx_img = *boost::any_cast<int>(&vm["img-zIndex"].value());
  int zIdx_det = *boost::any_cast<int>(&vm["det-zIndex"].value());
  float theta = *boost::any_cast<float>(&vm["rotation"].value());
  printf("Write to file, img-z:%d,det-z:%d,rotation:%f\n", zIdx_img, zIdx_det,
         theta);
  char ofname[128];
  sprintf(ofname, "matrix_img_%d_det_%d_rotation_%f.dat", zIdx_img, zIdx_det,
          theta);
  std::ofstream myFile(ofname, std::ios::out | std::ios::binary);
  if (!myFile) {
    std::cerr << "Error opening output file!\n";
    return -1;
  }
  for (int idx = 0; idx < NElement; idx++) {
    myFile.write((const char *)dataArray + (idx * sizeof(float)),
                 sizeof(float));
    if (!myFile) {
      std::cerr << "Error outputing to file!\n";
      return -1;
    }
  }
  myFile.close();
  auto stop3 = std::chrono::high_resolution_clock::now();
  duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop3 - start3);
  std::cout << "File output complete in: " << duration.count()
            << " microseconds.\n";
  //   for (int idx = 0; idx < NElement; idx++) {
  //     std::cout << dataArray[idx] << std::endl;
  //   }
  free(dataArray);
  std::cout << "Total time elapsed: "
            << (float)std::chrono::duration_cast<std::chrono::microseconds>(
                   stop3 - start1)
                       .count() /
                   1000000.0
            << " seconds.\n";
  printf("Done!\n");
  return 0;
}
