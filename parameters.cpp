#include "photodetector.h"
#include<iostream>
#include <cstdio>
#include<stdio.h>
#include "parameters.h"
#include "IniFile.h"
#include"include/rapidjson/document.h"
#include"include/rapidjson/filereadstream.h"


float* Parameters(float* parameter)
{
	float numDetector = pixelSiPM * numModuleT;
	float _float_RadiusDetector[10];

	_float_RadiusDetector[0] = RadiusMetalplate;
	_float_RadiusDetector[1] = RadiusDetectorFrontPlane;

	int totalnumDetector = 0;  ////4084 //4083 //4240
	for (int i = 1; i < numDetectorLayer + 1; ++i)
	{
		totalnumDetector += numDetector * numPanel;
	}

	for (int i = 1; i < numDetectorLayer + 1; ++i)
	{
		//0:collimator; 1~4:detector
		parameter[(i + 1) * 10] = numDetector;          // number of detectorY
		parameter[(i + 1) * 10 + 1] = numDetectorinAxial;
		parameter[(i + 1) * 10 + 2] = _float_RadiusDetector[1] + ((i - 1) + 0.5) * widthDetectorPixelX;          // Diameter
		parameter[(i + 1) * 10 + 3] = widthDetectorPixelX;         //ThicknessDetector
		parameter[(i + 1) * 10 + 4] = widthDetectorPixelY;   //width
		parameter[(i + 1) * 10 + 5] = widthDetectorPixelZ;
	}

	parameter[10] = numDetector* widthDetectorPixelY/ widthSlitZ;
	parameter[11] = numDetectorinAxial * widthDetectorPixelZ / widthSlitZ;
	parameter[12] = RadiusMetalplate;
	parameter[13] = thicknessMetalplateX;
	parameter[14] = widthSlitZ;
	parameter[15] = widthSlitZ;
	parameter[16] = slitGapZ;
	parameter[17] = numRotation;

	parameter[401] = thicknessCrystalX;
	parameter[402] = widthCrystalY;
	parameter[403] = widthCrystalZ;


	//parameter[400] = holeRatio;

	parameter[500] = microUnitX;
	parameter[501] = microUnitY;
	parameter[502] = microUnitZ;
	parameter[503] = microUnitPreviousLayer;

	parameter[600] = numPanel;
	parameter[700] = numDetector;

	parameter[900] = metalPlateMode; //0: no slit, 1:slit
	parameter[901] = crystalArrayMode; //0:1010; 1:1100

	/************************************** parameter *********************************************************/

	//Parameters
	parameter[1] = dimDetectorZ; // numDetector(axial);
	parameter[2] = numDetectorLayer;		// numDetectorLayer;
	parameter[3] = totalnumDetector;//4020 // 6030;		// totalnumDetector;
	parameter[4] = attenTungsten;
	parameter[5] = attenAir;    //AIR
	parameter[6] = attenGAGG;//GAGG
	parameter[7] = attenOpticalGlass;

	parameter[300] = numDetectorinAxial; // total numDetectorZ

	//image parameter
	parameter[1000] = dimImageZ; //67; //1; // 100; // 0;		// numImageSlice;

	parameter[1017] = numImageX;  // numImageVoxelX;
	parameter[1018] = numImageY;  // numImageVoxelY;
	parameter[1019] = numImageZ;  // numImageVoxelZ;

	parameter[1020] = widthImageX;  // widthImageVoxelX;
	parameter[1021] = widthImageY;  // widthImageVoxelY;
	parameter[1022] = widthImageZ;  // widthImageVoxelZ;

	/************************************** end *********************************************************/

	parameter[1023] = parameter[1017];// numPSFImageVoxelX;
	parameter[1024] = parameter[1018];//numPSFImageVoxelY;
	parameter[1025] = parameter[1019];//numPSFImageVoxelZ;
	parameter[1026] = parameter[1020];// widthPSFImageVoxelX;
	parameter[1027] = parameter[1021];//widthPSFImageVoxelY;
	parameter[1028] = parameter[1022];//widthPSFImageVoxelZ;
	parameter[1029] = widthSlitZ;
	parameter[1030] = slitGapZ;
	parameter[1031] = widthSlitY;
	parameter[1032] = slitGapY;

	return parameter;
}

string GetSysmatPath(){
	FILE* fp = fopen("../SysMatConfig/Parameters.json", "rb");
    if (!fp) {
        std::cerr << "Error: unable to open file"
                  << std::endl;
        return "";
    }
    char readBuffer[65536];
    rapidjson::FileReadStream is(fp, readBuffer,
                                 sizeof(readBuffer));
  
    // Parse the JSON document
    rapidjson::Document doc;
    doc.ParseStream(is);
  
    if (doc.HasParseError()) {
        std::cerr << "Error: failed to parse JSON document"
                  << std::endl;
        fclose(fp);
        return "";
    }
  
	string path = doc["sysmatPath"].GetString();
    return path;
}

int ReadInputJson()
{
	// IniFile par(ParameterFileName.c_str());
	// par.setSection("Parameters");
	// metalPlateMode = par.readInt("metalPlateMode", 0); //0: no metal plate, 1:metal plate with slits
	// crystalArrayMode = par.readInt("crystalArrayMode", 1); //0:1010; 1:1100
	// numRotation = par.readInt("numRotation", 1);

	// microUnitX = par.readDouble("microUnitX", 8);
	// microUnitY = par.readDouble("microUnitY", 24);
	// microUnitZ = par.readDouble("microUnitZ", 24);
	// microUnitPreviousLayer = par.readDouble("microUnitPreviousLayer", 8);

	// numPanel = par.readInt("numPanel", 1);
	// numModuleT = par.readInt("numModuleT", 4);
	// pixelSiPM = par.readInt("pixelSiPM", 8);
	// numDetectorLayer = par.readInt("numDetectorLayer", 24);
	// numDetectorinAxial = par.readInt("numDetectorinAxial", 100);

	// widthDetectorPixelX = par.readDouble("widthDetectorPixelX", 3.36);
	// widthDetectorPixelY = par.readDouble("widthDetectorPixelY", 3.36);
	// widthDetectorPixelZ = par.readDouble("widthDetectorPixelZ", 2);

	// thicknessCrystalX = par.readDouble("thicknessCrystalX", 3);
	// widthCrystalY = par.readDouble("widthCrystalY", 2);
	// widthCrystalZ = par.readDouble("widthCrystalZ", 2);

	// thicknessMetalplateX = par.readDouble("thicknessMetalplateX", 2);
	// widthSlitZ = par.readDouble("widthSlitZ", 1);
	// slitGapZ = par.readDouble("slitGapZ", 2);
	// widthSlitY = par.readDouble("widthSlitZ", 3);
	// slitGapY = par.readDouble("slitGapY", 5);

	// RadiusMetalplate = par.readDouble("RadiusMetalplate", 99);
	// RadiusDetectorFrontPlane = par.readDouble("RadiusDetector1st", 100);

	// attenGAGG = par.readDouble("attenGAGG", 0.475);
	// attenAir = par.readDouble("attenAir", 0.0);
	// attenOpticalGlass = par.readDouble("attenOpticalGlass", 0.0399);
	// attenTungsten = par.readDouble("attenTungsten", 3.6323);

	// numImageX = par.readDouble("numImageX", 180);
	// numImageY = par.readDouble("numImageY", 180);
	// numImageZ = par.readDouble("numImageZ", 200);
	// widthImageX = par.readDouble("widthImageX", 1.0);
	// widthImageY = par.readDouble("widthImageY", 1.0);
	// widthImageZ = par.readDouble("widthImageZ", 1.0);

	// dimImageZ = par.readInt("dimImageZ", 1);
	// dimDetectorZ = par.readInt("dimDetectorZ", 1);

	FILE* fp = fopen("../SysMatConfig/Parameters.json", "rb");
    if (!fp) {
        std::cerr << "Error: unable to open file"
                  << std::endl;
        return 1;
    }
    char readBuffer[65536];
    rapidjson::FileReadStream is(fp, readBuffer,
                                 sizeof(readBuffer));
  
    // Parse the JSON document
    rapidjson::Document doc;
    doc.ParseStream(is);
  
    if (doc.HasParseError()) {
        std::cerr << "Error: failed to parse JSON document"
                  << std::endl;
        fclose(fp);
        return 1;
    }
  
    metalPlateMode = doc["metalPlateMode"].GetInt(); //0: no metal plate, 1:metal plate with slits
	crystalArrayMode = doc["crystalArrayMode"].GetInt(); //0:1010; 1:1100
	numRotation = doc["numRotation"].GetInt();

	microUnitX = doc["microUnitX"].GetDouble();
	microUnitY = doc["microUnitY"].GetDouble();
	microUnitZ = doc["microUnitZ"].GetDouble();
	microUnitPreviousLayer = doc["microUnitPreviousLayer"].GetDouble();

	numPanel = doc["numPanel"].GetInt();
	numModuleT = doc["numModuleT"].GetInt();
	pixelSiPM = doc["pixelSiPM"].GetInt();
	numDetectorLayer = doc["numDetectorLayer"].GetInt();
	numDetectorinAxial = doc["numDetectorinAxial"].GetInt();

	widthDetectorPixelX = doc["widthDetectorPixelX"].GetDouble();
	widthDetectorPixelY = doc["widthDetectorPixelY"].GetDouble();
	widthDetectorPixelZ = doc["widthDetectorPixelZ"].GetDouble();

	thicknessCrystalX = doc["thicknessCrystalX"].GetDouble();
	widthCrystalY = doc["widthCrystalY"].GetDouble();
	widthCrystalZ = doc["widthCrystalZ"].GetDouble();

	thicknessMetalplateX = doc["thicknessMetalplateX"].GetDouble();
	widthSlitZ = doc["widthSlitZ"].GetDouble();
	slitGapZ = doc["slitGapZ"].GetDouble();
	widthSlitY = doc["widthSlitY"].GetDouble();
	slitGapY = doc["slitGapY"].GetDouble();

	RadiusMetalplate = doc["RadiusMetalplate"].GetDouble();
	RadiusDetectorFrontPlane = doc["RadiusDetectorFrontPlane"].GetDouble();

	attenGAGG = doc["attenGAGG"].GetDouble();
	attenAir = doc["attenAir"].GetDouble();
	attenOpticalGlass = doc["attenOpticalGlass"].GetDouble();
	attenTungsten = doc["attenTungsten"].GetDouble();

	numImageX = doc["numImageX"].GetDouble();
	numImageY = doc["numImageY"].GetDouble();
	numImageZ = doc["numImageZ"].GetDouble();
	widthImageX = doc["widthImageX"].GetDouble();
	widthImageY = doc["widthImageY"].GetDouble();
	widthImageZ = doc["widthImageZ"].GetDouble();

	dimImageZ = doc["dimImageZ"].GetInt();
	dimDetectorZ = doc["dimDetectorZ"].GetInt();
    // Close the file
    fclose(fp);

	cout<<"NUmImageZ = "<<numImageZ<< "NumimageX = "<< numImageX<<"NumimageY= "<<numImageY <<"\n";
	return 0;
}




