// C/C++ standard headers
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

// Non-standard headers
#include "rapidjson/document.h"
// Self-defined headers
#include "cmd_args.hh"
#include "config_parser.hh"
#include "myCUDA_functions.hh"

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
  //   int N_imgVoxelZ = jsonDoc["Image Volume"]["N Image Voxel Z"].GetInt();
  //   int N_imgSubX = jsonDoc["Image Volume"]["Voxel N Subdivision
  //   X"].GetInt(); int N_imgSubY = jsonDoc["Image Volume"]["Voxel N
  //   Subdivision Y"].GetInt(); int N_imgSubZ = jsonDoc["Image Volume"]["Voxel
  //   N Subdivision Z"].GetInt();

  //   return N_panel * N_moduleZ * N_blockX * N_blockY * N_cellX * N_cellY *
  //          N_xtalSegZ * N_imgVoxelX * N_imgVoxelY * N_imgVoxelZ;
  return N_blockX * N_blockY * N_cellX * N_cellY * N_imgX * N_imgY;
}

void calcBorderAndCoeff(std::vector<float> &borderX,
                        std::vector<float> &borderY,
                        std::vector<float> &borderZ,
                        std::vector<std::vector<float>> &h_coeffsMap,
                        const rapidjson::Document &jsonDoc) {
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
  printf("Block Crystal Configuration Pattern (%d x %d):\n", N_cellY, N_cellX);

  float coeffs[4];
  const rapidjson::Value &jsonCoeffs =
      jsonDoc["Detector"]["Attenuation Constants"];
  assert(jsonCoeffs.IsArray());
  auto coeffsArrObj = jsonCoeffs.GetArray();
  for (int idx = 0; idx < 4; idx++) {
    coeffs[idx] = coeffsArrObj[idx].GetFloat();
  }
  for (int i = 0; i < N_cellX; i++) {
    printf("\t");
    for (int j = 0; j < N_cellY; j++) {
      printf(" %d", pattern[i][j]);
    }
    printf("\n");
  }
  int N_subCellsX = N_cellX * N_blockX * (N_detSubX + 2);
  int N_subCellsY = N_cellY * N_blockY * (N_detSubY + 2);
  int N_subCellsZ = N_moduleZ * N_XtalSegZ * N_detSubZ;
  borderX.resize(N_subCellsX + 1);
  borderY.resize(N_subCellsY + 1);
  borderZ.resize(N_subCellsZ + 1);
  borderX[0] = 0;
  borderY[0] = 0;
  borderZ[0] = 0.0 - panelCenterZ;
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
  std::string configPath;
  rapidjson::Document RJDocument;

  int result = cmd_args_parser(argc, argv, configPath);
  if (result)
    return 1;
  std::cout << "Using config file:\t" << configPath << '\n';
  result = parse_config_json(configPath, RJDocument);
  if (result)
    return 1;
  std::cout << "Hello World!" << std::endl;

  int NElement = NElem(RJDocument);
  int arrSize = NElement * sizeof(float);
  float *dataArray = (float *)malloc(arrSize);
  printf("Matrix N Elements:\t%d\nMatrix Data Size:\t%d (Bytes)\n", NElement,
         arrSize);

  std::vector<float> vecBorderX, vecBorderY, vecBorderZ;
  std::vector<std::vector<float>> h_coeffsMap;
  calcBorderAndCoeff(vecBorderX, vecBorderY, vecBorderZ, h_coeffsMap,
                     RJDocument);

  //   for (auto v : h_coeffsMap) {
  //     for (auto elem : v) {
  //       std::cout << std::setw(6) << elem << " ";
  //     }
  //     std::cout << std::endl;
  //   }

  sysMatCalc_1z_1z(dataArray, vecBorderX, vecBorderY, vecBorderZ,
                   h_coeffsMap, RJDocument, NElement);

  std::ofstream myFile("array.dat", std::ios::out | std::ios::binary);
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
  //   for (int idx = 0; idx < NElement; idx++) {
  //     std::cout << dataArray[idx] << std::endl;
  //   }
  free(dataArray);
  //   delete[] borderX;
  //   delete[] borderY;
  //   delete[] borderZ;
  //   delete[] h_coeffsMap;
  return 0;
}
