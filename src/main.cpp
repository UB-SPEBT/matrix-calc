// C/C++ standard headers
#include <iostream>
#include <iomanip>
#include <fstream>

// Non-standard headers
#include "rapidjson/document.h"
// Self-defined headers
#include "cmd_args.hh"
#include "config_parser.hh"
#include "myCUDA_functions.hh"

int NElem(const rapidjson::Document &jsonDoc)
{
    int NPanel = jsonDoc["Detector"]["N Panels"].GetInt();
    int NModule = jsonDoc["Detector"]["N Modules Axial"].GetInt();
    int NXtalSeg = jsonDoc["Detector"]["N Crystal Segments"].GetInt();
    int NCellX = jsonDoc["Detector"]["Module"]["N Cells X"].GetInt();
    int NCellY = jsonDoc["Detector"]["Module"]["N Cells Y"].GetInt();
    int NCellZ = jsonDoc["Detector"]["Module"]["N Cells Z"].GetInt();
    return NPanel * NModule * NCellX * NCellY * NCellZ * NXtalSeg;
}

int main(int argc, char **argv)
{
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
    printf("N Total Matrix Elements:\t%d\nArray Size:\t%d\n", NElement, arrSize);

    float dataArray[NElement];
    sysMatCalc_1z_1z(dataArray, RJDocument, NElement);

    std::ofstream myFile("array.dat", std::ios::out | std::ios::binary);
    if (!myFile)
    {
        std::cerr << "Error opening output file!\n";
        return -1;
    }
    for (int idx = 0; idx < NElement; idx++)
    {
        myFile.write((const char *)dataArray + (idx * sizeof(float)),
                     sizeof(float));
        if (!myFile)
        {
            std::cerr << "Error outputing to file!\n";
            return -1;
        }
    }
    myFile.close();
    // for (int idx = 0; idx < NElement; idx++)
    // {
    //     std::cout << dataArray[idx] << std::endl;
    // }
    // free(dataArray);
    return 0;
}
