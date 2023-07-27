#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>

#include "photodetector.h"
#include <fstream>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define max(a, b) ((a >= b) ? a : b)
#define min(a, b) ((a <= b) ? a : b)

using namespace std;

// choose the gpu
__global__ void photodetectorCudaMe(float *dst,
									float *deviceparameter,
									int numProjectionSingle,
									int numImagebin)

{

	int pattern[16][16] = {{1,0,1,1,1,1,0,1,0,0,0,1,1,0,0,0},
	{0,1,0,0,0,0,1,0,1,1,1,0,0,1,1,1},
	{1,0,1,1,1,1,0,1,0,0,0,1,1,0,0,0},
	{1,0,1,1,1,1,0,1,0,0,0,1,1,0,0,0},
	{1,0,1,1,1,1,0,1,0,0,0,1,1,0,0,0},
	{1,0,1,1,1,1,0,1,0,0,0,1,1,0,0,0},
	{0,1,0,0,0,0,1,0,1,1,1,0,0,1,1,1},
	{1,0,1,1,1,1,0,1,0,0,0,1,1,0,0,0},
	{0,1,0,0,0,0,1,0,1,1,1,0,0,1,1,1},
	{0,1,0,0,0,0,1,0,1,1,1,0,0,1,1,1},
	{0,1,0,0,0,0,1,0,1,1,1,0,0,1,1,1},
	{1,0,1,1,1,1,0,1,0,0,0,1,1,0,0,0},
	{1,0,1,1,1,1,0,1,0,0,0,1,1,0,0,0},
	{0,1,0,0,0,0,1,0,1,1,1,0,0,1,1,1},
	{0,1,0,0,0,0,1,0,1,1,1,0,0,1,1,1},
	{0,1,0,0,0,0,1,0,1,1,1,0,0,1,1,1}};


	int metalPlateMode = (int)deviceparameter[900];
	int crystalArrayMode = (int)deviceparameter[901];

	float _float_numDetectoraxial = deviceparameter[1];
	float _float_numDetectorLayer = deviceparameter[2];

	int numDetectorY[100], numDetectorZ[100];
	float detectorYWidth[100], detectorZWidth[100];
	float detectorXWidth[100];
	float detectorXpoint[100];

	for (int i = 0; i < (int)_float_numDetectorLayer + 1; i++)
	{
		// 0:collimator; 1~4:detector
		numDetectorY[i] = deviceparameter[(i + 1) * 10];				 // number of detector
		numDetectorZ[i] = deviceparameter[(i + 1) * 10 + 1];			 // number of detector
		detectorXpoint[i] = deviceparameter[(i + 1) * 10 + 2];	 // Diameter
		detectorXWidth[i] = deviceparameter[(i + 1) * 10 + 3]; // ThicknessDetector
		detectorYWidth[i] = deviceparameter[(i + 1) * 10 + 4];	 // width
		detectorZWidth[i] = deviceparameter[(i + 1) * 10 + 5];	 // width
	}

	float CrystalXWidth = deviceparameter[401];
	float CrystalYWidth = deviceparameter[402];
	float CrystalZWidth = deviceparameter[403];

	float _float_tungsten_attenuation_coeff = deviceparameter[4]; // 1.882 * 1.93// tungsten_attenuation_coeff

	float _float_detector_attenuation_coeff[11];
	_float_detector_attenuation_coeff[0] = deviceparameter[5]; // 0 //AIR
	_float_detector_attenuation_coeff[1] = deviceparameter[6]; // 0.475 //GAGG
	_float_detector_attenuation_coeff[2] = deviceparameter[7]; // OpticalGlass

	float numDetectorinAxial = deviceparameter[300];
	float idxDetectorZ = deviceparameter[301];

	float idxImageZ_1 = deviceparameter[302];

	float slitGap = deviceparameter[16];
	float _float_numRotation = deviceparameter[17];//numRotation;
	float _float_idxrotation = deviceparameter[18];//idxRotation
	float RotationAngle = _float_idxrotation * 0.03358736; //tan-1(3.36/100 / numRotations)

	float microUnitX = deviceparameter[500];
	float microUnitY = deviceparameter[501];
	float microUnitZ = deviceparameter[502];
	float microUnitPreviousLayer = deviceparameter[503];

	float numPanel = deviceparameter[600]; //6
	float numDetector = deviceparameter[700]; //32 = pixelSiPM * numModuleT

	float _float_numImageVoxelX = deviceparameter[1017];
	float _float_numImageVoxelY = deviceparameter[1018];
	float _float_numImageVoxelZ = deviceparameter[1019];

	float _float_numImageSlice = deviceparameter[1000];

	float _float_widthImageVoxelX = deviceparameter[1020];
	float _float_widthImageVoxelY = deviceparameter[1021];
	float _float_widthImageVoxelZ = deviceparameter[1022];

	float _float_numPSFImageVoxelX = deviceparameter[1023];
	float _float_numPSFImageVoxelY = deviceparameter[1024];
	float _float_numPSFImageVoxelZ = deviceparameter[1025];

	float _float_widthPSFImageVoxelX = deviceparameter[1026];
	float _float_widthPSFImageVoxelY = deviceparameter[1027];
	float _float_widthPSFImageVoxelZ = deviceparameter[1028];
	float widthSlitZ = deviceparameter[1029];
	float slitGapZ = deviceparameter[1030];
	float widthSlitY = deviceparameter[1031];
	float slitGapY = deviceparameter[1032];
	
	float _float_dimDetectorZ = deviceparameter[1]; // numDetector(axial);

	int numDetectorLayer = (int)floor(_float_numDetectorLayer);
	int dimDetectorZ = (int)floor(_float_dimDetectorZ);

	int totalnumDetectorZ = (int)floor(numDetectorinAxial);
	int startidxDetectorZ = (int)floor(idxDetectorZ);
	int startidxImageZ = (int)floor(idxImageZ_1);

	int numImageVoxelX = (int)floor(_float_numImageVoxelX);
	int numImageVoxelY = (int)floor(_float_numImageVoxelY);
	int numImageVoxelZ = (int)floor(_float_numImageVoxelZ);
	int numImageSlice = (int)floor(_float_numImageSlice);

	int numPSFImageVoxelX = (int)floor(_float_numPSFImageVoxelX);
	int numPSFImageVoxelY = (int)floor(_float_numPSFImageVoxelY);
	int numPSFImageVoxelZ = (int)floor(_float_numPSFImageVoxelZ);

	bool inside = false;

	// GPU thread
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < 0 || row > numProjectionSingle - 1)
	{
		return;
	}
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (col < 0 || col > numImagebin - 1)
	{
		return;
	}

	// DRF
	int dstIndex = row * numImagebin + col;

	// module index
	int numPanels = (int)numPanel;
	int  numDetectorSinglePanel  = (int)numDetector;

	int idxDetectorLayerX = row / ( numDetectorSinglePanel  * numPanels); // radial, X coordinate
	row = row % ( numDetectorSinglePanel  * numPanels);
	int idxDetectorY = row;// / dimDetectorZ; // tangential, Y coordinate
	// idxZ in totalnumDetectorZ

	int idxDetectorlayerZ = startidxDetectorZ;
	int idxImageZ = startidxImageZ;

	col = col % (numPSFImageVoxelY * numPSFImageVoxelX);
	int idxImageY = col / numPSFImageVoxelX;
	int idxImageX = col % numPSFImageVoxelX;

	const float divideX = (int)microUnitX, divideY = (int)microUnitY, divideZ = (int)microUnitZ;
	int MicroNum = (int)microUnitPreviousLayer;

	float xImage = (idxImageX - _float_numImageVoxelX / 2.0 + 0.5) * _float_widthPSFImageVoxelX;
	float yImage = (idxImageY - _float_numImageVoxelY / 2.0 + 0.5) * _float_widthPSFImageVoxelY;
	float zImage = (idxImageZ - _float_numImageVoxelZ / 2.0 + 0.5) * _float_widthPSFImageVoxelZ;

	float xImage_rot = xImage*cos(RotationAngle) - yImage*sin(RotationAngle);
	float yImage_rot = xImage*sin(RotationAngle) + yImage*cos(RotationAngle);
	
	xImage = xImage_rot; yImage = yImage_rot;

	int idxPanel = idxDetectorY /  numDetectorSinglePanel ;
	idxDetectorY = idxDetectorY -  numDetectorSinglePanel  * idxPanel; 

	float moduleRot = 2 * M_PI / numPanel * idxPanel;

	float _float_translationLength = deviceparameter[800];
	float _float_numTranslation = deviceparameter[801];
	float _float_idxTranslationZ = deviceparameter[803];

	float attenuation_coefficient_each, attenuation_coefficient_final, attenuation_dist, attenuation_distFINAL,
		distancesq, COSangle, solid_angle;

    //dst[idxDetectorLayerX][idxPanel][idxDetectorY][idxImageX][idxImageY] = 0;
	dst[dstIndex] = 0.0;

	float detection_eff_each;
	float CrystalR;
	float zDetector0 = (idxDetectorlayerZ - totalnumDetectorZ / 2.0 + 0.5) * detectorZWidth[idxDetectorLayerX + 1];

	// CENTER OF THE PANEL
	float detectorX0, detectorY0;
	float detectorY00 = (-0.5 *  numDetectorSinglePanel  + idxDetectorY + 0.5) * detectorYWidth[idxDetectorLayerX + 1];
	float xDetector, yDetector, zDetector;

	float MicroCOSangle, length;

	float xMicro = 0, yMicro = 0, zMicro = 0;
	float xMicro00, yMicro00, zMicro00;
	int idxDetectorYmicro, idxDetectorZmicro;
	bool isInCrystalY, isInCrystalX, isDetectorZmicro, isMosaicXYmicro;

	float coeff;
	int index;
	//printf("xImage,yImage,zImage = %f,%f,%f\n",xImage,yImage,zImage);
	bool coeff_crossing;
	float diameter = 180.0;
	if (xImage * xImage + yImage * yImage > diameter * diameter / 4.0)
	{
		dst[dstIndex] = 0;
	}
	else
	{
		for (int NumZ = 0; NumZ < divideZ; NumZ++)
		{
			for (int NumY = 0; NumY < divideY; NumY++)
			{
				for (int NumX = 0; NumX < divideX; NumX++)
				{
					detection_eff_each = 1;

					//Calculate the center coordiantes for the microunit
					float start_crystalX =  detectorXpoint[idxDetectorLayerX + 1] - (CrystalXWidth/2);
					float start_crystalY =  detectorY00 - (CrystalYWidth/2);
					float start_crystalZ =  zDetector0 - (CrystalZWidth/2);
					
					CrystalR = start_crystalX + ((NumX + 0.5) * (CrystalXWidth/divideX));
					zDetector = start_crystalZ + ((NumZ + 0.5) * (CrystalZWidth/divideZ));
					detectorX0 = CrystalR;
					detectorY0 = start_crystalY + ((NumY + 0.5) * (CrystalYWidth/divideY));
					//detectorY0 = detectorY00;// + ((float)(NumY + 0.5) / (float)divideY - 0.5) * CrystalYWidth;

					//Rotate the point
					xDetector = cos(moduleRot) * detectorX0 - sin(moduleRot) * detectorY0;
					yDetector = sin(moduleRot) * detectorX0 + cos(moduleRot) * detectorY0;

					//(xDetector,yDetector,zDetector) - center of the MicroUnit

					//printf("xDetector,yDetector,zDetector,startZ = %f,%f,%f,%f\n",xDetector,yDetector,zDetector,start_crystalZ);

					float detectorX00 = CrystalR;

					float x11 = cos(moduleRot) * detectorX00 - sin(moduleRot) * 0.0;
					float y11 = sin(moduleRot) * detectorX00 + cos(moduleRot) * 0.0;
					float z11 = 0.0;

					//printf("x11,y11,z11 = %f,%f,%f\n",x11,y11,z11);

					distancesq = (yDetector - yImage) * (yDetector - yImage) + (xDetector - xImage) * (xDetector - xImage) + (zDetector - zImage) * (zDetector - zImage);

					// angle between module plane and LOR
					float a = sqrt(distancesq);
					float b = sqrt(x11 * x11 + y11 * y11 + z11 * z11);
					float c = (xDetector - xImage) * x11 + (yDetector - yImage) * y11 + (zDetector - zImage) * z11;
					COSangle = c / (a * b);

					//Calculate Solid Angle
					solid_angle = (CrystalYWidth * CrystalZWidth / (divideY * divideZ)) / (4 * M_PI * distancesq) * COSangle;


					attenuation_dist = 0;
					for (int m = 0; m <= idxDetectorLayerX + 1; m++) 
					{
						for (int idxMicro = 0; idxMicro < MicroNum; idxMicro++)
						{
							coeff = 0;
							//Calculate center of the microunit
							float M_start_X =  detectorXpoint[m] - (detectorXWidth[m]/2);
							float MicroR = M_start_X + detectorXWidth[m] * (float)(idxMicro + 0.5) / (float)MicroNum;

							
							if (m == idxDetectorLayerX + 1)
							{
								MicroR = M_start_X + detectorXWidth[m] * (float)(NumX) / (float)divideX * (float)(idxMicro + 0.5) / (float)MicroNum;

							}

							//float MicroR = detectorXpoint[m];// + ((idxMicro + 0.5) / (float)MicroNum - 0.5) * detectorXWidth[m];
							// if (m == idxDetectorLayerX + 1)
							// {
							// 	MicroR = detectorXpoint[m];// - 0.5 * detectorXWidth[m] + ((idxMicro + 0.5) / (float)MicroNum) * detectorXWidth[m] * (float)(NumX) / (float)divideX;
							// }
							// zMicro = zImage; //2D detector

							float xMicro0 = MicroR;
							float yMicro0 = 0.;
							float zMicro0 = 0.;

							float xMicro1 = cos(moduleRot) * xMicro0 - sin(moduleRot) * yMicro0;
							float yMicro1 = sin(moduleRot) * xMicro0 + cos(moduleRot) * yMicro0;
							float zMicro1 = zMicro0;

							// dst[dstIndex] = xMicro1;

							float x0 = xImage, y0 = yImage, z0 = zImage, x2 = xDetector, y2 = yDetector, z2 = zDetector, x1 = xMicro1, y1 = yMicro1, z1 = zMicro1;

							float temp = (y2 - y0) * y1 + x1 * (x2 - x0);
							float temp2 = y1 * (y1 - y0) * (x2 - x0) + x1 * x1 * (x2 - x0) + x0 * y1 * (y2 - y0);

							if (abs(x2 - x0) < 0.001)
							{
								xMicro = x0;
								yMicro = y1 - x1 * (x2 - x1) / y1;
								zMicro = (z2 - z0) * (yMicro - y0) / (y2 - y0) + z0;
							}
							else if (abs(y2 - y0) < 0.001)
							{
								xMicro = y1 * (y1 - y0) / x1 + x1;
								yMicro = y0;
								zMicro = (z2 - z0) * (xMicro - x0) / (x2 - x0) + z0;
							}
							else
							{
								xMicro = temp2 / temp;
								// float yMicro = -x1 * (xMicro - x1) / y1 + y1;
								yMicro = (y2 - y0) / (x2 - x0) * (xMicro - x0) + y0;
								zMicro = (xMicro - x0) / (x2 - x0) * (z2 - z0) + z0;
							}

							// COSangle
							a = sqrt((yMicro - y0) * (yMicro - y0) + (xMicro - x0) * (xMicro - x0) + (zMicro - z0) * (zMicro - z0));
							b = sqrt(x1 * x1 + y1 * y1 + z1 * z1);
							c = (xMicro - x0) * x1 + (yMicro - y0) * y1 + (zMicro - z0) * z1;
							MicroCOSangle = c / (a * b);

							// length
							length = detectorXWidth[m] / MicroCOSangle / (float)MicroNum;
							if (m == idxDetectorLayerX + 1)
							{
								length = detectorXWidth[m]  * (float)(NumX) / (float)divideX / MicroCOSangle / (float)MicroNum;
							}

							xMicro00 = cos(-moduleRot) * xMicro - sin(-moduleRot) * yMicro;
							yMicro00 = sin(-moduleRot) * xMicro + cos(-moduleRot) * yMicro;
							zMicro00 = zMicro;

							float idxtemp = yMicro00 + numDetectorY[m] / 2 * detectorYWidth[m];
							idxDetectorYmicro = floor(idxtemp / detectorYWidth[m]);

							idxDetectorZmicro = floor((zMicro + (float)numDetectorZ[m] * detectorZWidth[m] / 2.0) / detectorZWidth[m]);

							if ((yMicro00 < -1.0 * numDetectorY[idxDetectorLayerX + 1] / 2.0 * detectorYWidth[idxDetectorLayerX + 1]) ||
								(yMicro00 > numDetectorY[idxDetectorLayerX + 1] / 2.0 * detectorYWidth[idxDetectorLayerX + 1]) ||
								(zMicro00 < -1.0 * (float)numDetectorZ[idxDetectorLayerX + 1] / 2.0 * detectorZWidth[idxDetectorLayerX + 1]) ||
								(zMicro00 > (float)numDetectorZ[idxDetectorLayerX + 1] / 2.0 * detectorZWidth[idxDetectorLayerX + 1]))
							{
								coeff = 10000.0;
							}
							else
							{
								coeff = 0;

								isInCrystalY = abs(idxtemp - (idxDetectorYmicro + 0.5) * detectorYWidth[m]) <= CrystalYWidth / 2.0;
								isInCrystalX = abs(xMicro00 - detectorXpoint[m]) <= CrystalXWidth / 2.0;

								isDetectorZmicro = idxDetectorZmicro % 2;
								isMosaicXYmicro;

								if (crystalArrayMode == 0) // 1010
								{
									isMosaicXYmicro = (idxDetectorYmicro + (m + 1)) % 2;
								}
								if (crystalArrayMode == 1) // 1100
								{
									isMosaicXYmicro = (((idxDetectorYmicro % 4 == 0 || idxDetectorYmicro % 4 == 1) && ((m + 1) % 4 == 0 || (m + 1) % 4 == 1)) ||
													   ((idxDetectorYmicro % 4 == 2 || idxDetectorYmicro % 4 == 3) && ((m + 1) % 4 == 2 || (m + 1) % 4 == 3)));
								}

								if( crystalArrayMode == 2)
								{
									//printf("inside");
									int n = (idxDetectorYmicro)%16;
									isMosaicXYmicro = pattern[(m+1)%16][n];
								}

								if (isInCrystalX && isInCrystalY)
								{
									if (isMosaicXYmicro == isDetectorZmicro)
									{
										coeff = _float_detector_attenuation_coeff[1];
									}
									else
									{
										coeff = _float_detector_attenuation_coeff[2];
									}
								}

								if (m == 0) // colli
								{

									int test1 = -1;
									// translation
									// zMicro00 = zMicro00 + _float_translationLength / _float_numTranslation * _float_idxTranslationZ;
									// zMicro00 = zMicro00 + (float)numDetectorZ[m] * detectorZWidth[m] / 2.0;
									// idxDetectorZmicro = floor(zMicro00 / slitGap);
									// coeff_crossing = abs(zMicro00 - (idxDetectorZmicro + 0.5) * slitGap) < 0.5 * detectorZWidth[m]; // translation
									// coeff = _float_tungsten_attenuation_coeff * (1 - coeff_crossing) * metalPlateMode;
									int temp1 = floor(floor((numDetector*detectorYWidth[1])/slitGapY)/2);
									int temp2 = floor(floor(CrystalZWidth/slitGapZ)/2);
									for (int idxtest1 = 0; idxtest1 <= temp2; idxtest1++) // aperture number (transaxial): 2 * 9 + 1 = 19
									{
										if (fabs(fabs(zMicro00) - idxtest1 * slitGapZ) <  (widthSlitZ/2)) //aperture size: 4 mm
										{
											test1 = 1;
										}
									}
									for (int idxtest1 = 0; idxtest1 <= temp1; idxtest1++) // aperture number (axial): 2 * 6 + 1 = 13
									{
										if (fabs(fabs(yMicro00) - idxtest1 * slitGapY) < (widthSlitY/2) && test1 == 1) //aperture size: 4 mm
										{
											coeff = 0;
											test1 = 2;
										}
									}

									if(test1 == -1 || test1 == 1)
									{
										coeff = _float_tungsten_attenuation_coeff * metalPlateMode;
									}
								}
							}

							if (m == numDetectorLayer)
							{
								coeff = _float_detector_attenuation_coeff[1];
							}
							attenuation_dist = attenuation_dist + coeff * length;
						}

					} // idxDetectorLayer
					detection_eff_each = detection_eff_each * exp(-attenuation_dist);

					attenuation_coefficient_each = 0;
					bool isMosaic;
					if (crystalArrayMode == 0) // 1010
					{
						isMosaic = (idxDetectorY + idxDetectorLayerX) % 2;
					}
					if (crystalArrayMode == 1) // 1100
					{
						isMosaic = ((idxDetectorY % 4 == 0 || idxDetectorY % 4 == 1) && (idxDetectorLayerX % 4 == 0 || idxDetectorLayerX % 4 == 1)) ||
								   ((idxDetectorY % 4 == 2 || idxDetectorY % 4 == 3) && (idxDetectorLayerX % 4 == 2 || idxDetectorLayerX % 4 == 3));
					}

					if( crystalArrayMode == 2)
					{
						//printf("inside");
						int n = (idxDetectorY)%16;
						isMosaic = pattern[idxDetectorLayerX%16][n];
					}
					bool isDetectorLayerZ = idxDetectorlayerZ % 2;
					if (isMosaic == isDetectorLayerZ)
					{
						attenuation_coefficient_each = _float_detector_attenuation_coeff[1];
					}

					if (idxDetectorLayerX + 1 == numDetectorLayer)
					{
						attenuation_coefficient_each = _float_detector_attenuation_coeff[1];
					}
					attenuation_distFINAL = (detectorXWidth[idxDetectorLayerX + 1] / divideX) / COSangle * attenuation_coefficient_each;
                    //dst[idxImageX][idxImageY][idxDetectorLayerX][idxDetectorY] += detection_eff_each * solid_angle * (1 - exp(-attenuation_distFINAL)); 
					dst[dstIndex] += detection_eff_each * solid_angle * (1 - exp(-attenuation_distFINAL));
				} // divideX
			}	  // divideY
		}		  // divideZ
		// dst[dstIndex] = zMicro00;
	} // diameter=180mm
}

int photodetector(float *parameter, float *dst)
{

	float _float_totalnumDetector = parameter[3];
	float _float_numDetectorZ = parameter[1];
    float _float_numDetectorX = parameter[2];
    float _float_numDetectorY = parameter[700];
	float _float_numImageSliceZ = parameter[1000];
	float _float_numPSFImageVoxelX = parameter[1017];
	float _float_numPSFImageVoxelY = parameter[1018];
	float _float_numPSFImageVoxelZ = parameter[1019];

	int totalnumDetector = (int)floor(_float_totalnumDetector);
	int numDetectorZ = (int)floor(_float_numDetectorZ);

	int numImageSliceZ = (int)floor(_float_numImageSliceZ);
	int numPSFImageVoxelX = (int)floor(_float_numPSFImageVoxelX);
	int numPSFImageVoxelY = (int)floor(_float_numPSFImageVoxelY);
	int numPSFImageVoxelZ = (int)floor(_float_numPSFImageVoxelZ);

	int numProjectionSingle = totalnumDetector;
	int numImagebin = numPSFImageVoxelX * numPSFImageVoxelY;
    int numDetectorX = (int)floor(_float_numDetectorX);
    int numDetectorY = (int)floor(_float_numDetectorY);

	float* deviceMatrix, * deviceparameter;
	cudaMalloc(&deviceMatrix, sizeof(float) * numProjectionSingle * numImagebin);
	cudaMemset(deviceMatrix, 1, sizeof(float) * numProjectionSingle * numImagebin);
	cudaMalloc(&deviceparameter, sizeof(float) * 1280);
	cudaMemcpy(deviceparameter, parameter, sizeof(float) * 1280, cudaMemcpyHostToDevice);

	dim3 blockSize(4, 64);
	dim3 gridSize((numProjectionSingle + 3) / blockSize.x, (numImagebin + 63) / blockSize.y);

	cout << "########################" << endl;
	cout << "double check for GPU calculation" << endl;
	cout << "numProjectionSingle = " << numProjectionSingle << endl;
	cout << "numImagebin = " << numImagebin << endl;
	cout << "gridSize.x = " << gridSize.x << endl;
	cout << "gridSize.y = " << gridSize.y << endl;
	cout << "########################" << endl;

	photodetectorCudaMe <<<gridSize, blockSize >>> (
		deviceMatrix,
		deviceparameter,
		numProjectionSingle,
		numImagebin);

	// cudaThreadSynchronize();
	float* test = new float[1280];
	cudaMemcpy(dst, deviceMatrix, sizeof(float) * numProjectionSingle * numImagebin, cudaMemcpyDeviceToHost);
	cudaMemcpy(test, deviceparameter, sizeof(float) * 1280, cudaMemcpyDeviceToHost);

	cout << "########################" << endl;
	cout << "parameter from device" << endl;
	for (int i = 0; i <= 100; i++)
	{
		cout << "test[" << i << "]= " << dst[i] << endl;
	}
	cout << "########################" << endl;

	cudaFree(deviceparameter);
	cudaFree(deviceMatrix);

	return (numImagebin);
}
