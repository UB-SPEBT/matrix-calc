// Define the configuration structure used by CUDA kernel function.
#include "rapidjson/document.h"

// This structure should only contain parameters need by the matrix calculation kernel function
struct configList
{
    // int metalPlateMode = (int)deviceparameter[900];
    // int crystalArrayMode = (int)deviceparameter[901];

    float cellLenX, cellLenY, cellLenZ;
    float attenuation_coeff_tungsten, attenuation_coeff_GAGG, attenuation_coeff_glass;
    float slitGap;

    int NPanel, NModule, NXtalSeg;
    int NCellX, NCellY, NCellZ;
    int NSubdivDetX, NSubdivDetY, NSubdivDetZ;
    int NImgX, NImgY, NImgZ;
    int NSubdivImgX, NSubdivImgY, NSubdivImgZ;
};

void initConfList(struct configList *conf, const rapidjson::Document &jsonDoc)
{
    conf->NPanel = jsonDoc["Detector"]["N Panels"].GetInt();
    conf->NModule = jsonDoc["Detector"]["N Modules Axial"].GetInt();
    conf->NXtalSeg = jsonDoc["Detector"]["N Crystal Segments"].GetInt();
    conf->NCellX = jsonDoc["Detector"]["Module"]["N Cells X"].GetInt();
    conf->NCellY = jsonDoc["Detector"]["Module"]["N Cells Y"].GetInt();
    conf->NCellZ = jsonDoc["Detector"]["Module"]["N Cells Z"].GetInt();

    conf->cellLenX = jsonDoc["Detector"]["Module"]["Cell Length X"].GetFloat();
    conf->cellLenY = jsonDoc["Detector"]["Module"]["Cell Length Y"].GetFloat();
    conf->cellLenZ = jsonDoc["Detector"]["Module"]["Cell Length Z"].GetFloat();
};