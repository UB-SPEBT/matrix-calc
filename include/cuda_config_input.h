// Define the configuration structure used by CUDA kernel function.

#include "rapidjson/document.h"
#include <math.h>
#include <vector>
#define PI 3.14159265359
// This structure should only contain parameters need by the matrix calculation
// kernel function
struct configList {
  // int metalPlateMode = (int)deviceparameter[900];
  // int crystalArrayMode = (int)deviceparameter[901];

  int N_panel, N_moduleZ, N_XtalSegZ, N_detSubX, N_detSubY, N_detSubZ, N_blockX,
      N_blockY, N_cellX, N_cellY, N_subCellsX, N_subCellsY, N_subCellsZ;
  int N_imgX, N_imgY, N_imgZ, N_imgSubX, N_imgSubY, N_imgSubZ;
  float attenuation_coeff_tungsten, attenuation_coeff_GAGG,
      attenuation_coeff_glass;
  float cellLengthX, cellLengthY, XtalLengthX, XtalLengthY, XtalLengthZ;
  float imgLenX, imgLenY, imgLenZ, imgSubLenX, imgSubLenY, imgSubLenZ;
  float imgCenterX, imgCenterY, imgCenterZ;
  float theta;
  float fov_radius, panelW;
  int N_bordersX, N_bordersY, N_bordersZ;
  long total_N_subCells, total_N_subImg, total_N_subDet;
  int zIdx_img, zIdx_det;
};

inline void initConfList(struct configList *conf,
                         const rapidjson::Document &jsonDoc, float theta,
                         std::vector<int> &N_borders, int total_N_subDet,
                         int zIdx_img, int zIdx_det) {
  conf->N_panel = jsonDoc["Detector"]["N Panels"].GetInt();
  conf->N_moduleZ = jsonDoc["Detector"]["N Modules Axial"].GetInt();
  conf->N_XtalSegZ = jsonDoc["Detector"]["N Crystal Segments"].GetInt();
  conf->N_detSubX = jsonDoc["Detector"]["Detector N Subdivision X"].GetInt();
  conf->N_detSubY = jsonDoc["Detector"]["Detector N Subdivision Y"].GetInt();
  conf->N_detSubZ = jsonDoc["Detector"]["Detector N Subdivision Z"].GetInt();
  conf->cellLengthX = jsonDoc["Detector"]["Cell Length X"].GetFloat();
  conf->cellLengthY = jsonDoc["Detector"]["Cell Length Y"].GetFloat();
  conf->XtalLengthX = jsonDoc["Detector"]["Crystal Length X"].GetFloat();
  conf->XtalLengthY = jsonDoc["Detector"]["Crystal Length Y"].GetFloat();
  conf->XtalLengthZ = jsonDoc["Detector"]["Crystal Length Z"].GetFloat();
  // Module configuration
  conf->N_blockX = jsonDoc["Detector"]["Module"]["N Block X"].GetInt();
  conf->N_blockY = jsonDoc["Detector"]["Module"]["N Block Y"].GetInt();
  conf->N_cellX = jsonDoc["Detector"]["Module"]["Block"]["N Cells X"].GetInt();
  conf->N_cellY = jsonDoc["Detector"]["Module"]["Block"]["N Cells Y"].GetInt();
  conf->N_subCellsX = conf->N_cellX * conf->N_blockX * (conf->N_detSubX + 2);
  conf->N_subCellsY = conf->N_cellY * conf->N_blockY * (conf->N_detSubY + 2);
  conf->N_subCellsZ = conf->N_moduleZ * conf->N_XtalSegZ * conf->N_detSubZ;
  conf->fov_radius = jsonDoc["Detector"]["FOV Radius"].GetFloat();
  // Image configuration
  conf->N_imgX = jsonDoc["Image Volume"]["N Image Voxel X"].GetInt();
  conf->N_imgY = jsonDoc["Image Volume"]["N Image Voxel Y"].GetInt();
  conf->N_imgZ = jsonDoc["Image Volume"]["N Image Voxel Z"].GetInt();
  conf->N_imgSubX = jsonDoc["Image Volume"]["Voxel N Subdivision X"].GetInt();
  conf->N_imgSubY = jsonDoc["Image Volume"]["Voxel N Subdivision Y"].GetInt();
  conf->N_imgSubZ = jsonDoc["Image Volume"]["Voxel N Subdivision Z"].GetInt();

  conf->imgLenX = jsonDoc["Image Volume"]["Image Voxel Length X"].GetFloat();
  conf->imgLenY = jsonDoc["Image Volume"]["Image Voxel Length Y"].GetFloat();
  conf->imgLenZ = jsonDoc["Image Volume"]["Image Voxel Length Z"].GetFloat();
  conf->imgSubLenX = conf->imgLenX / conf->N_imgSubX;
  conf->imgSubLenY = conf->imgLenY / conf->N_imgSubY;
  conf->imgSubLenZ = conf->imgLenZ / conf->N_imgSubZ;
  conf->theta = theta;
  conf->imgCenterX = conf->imgLenX * conf->N_imgX * 0.5;
  conf->imgCenterY = conf->imgLenY * conf->N_imgY * 0.5;
  conf->imgCenterZ = conf->imgLenZ * conf->N_imgZ * 0.5;
  conf->panelW = conf->cellLengthX * conf->N_cellX;
  conf->N_bordersX = N_borders[0];
  conf->N_bordersY = N_borders[1];
  conf->N_bordersZ = N_borders[2];
  conf->total_N_subCells =
      conf->N_subCellsX * conf->N_subCellsY * conf->N_subCellsZ;
  conf->total_N_subImg = conf->N_imgX * conf->N_imgY * conf->N_imgSubX *
                         conf->N_imgSubY * conf->N_imgSubZ;
  conf->total_N_subDet = total_N_subDet;
  conf->zIdx_det = zIdx_det;
  conf->zIdx_img = zIdx_img;
};