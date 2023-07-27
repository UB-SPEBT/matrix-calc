// PhotoDetector.cpp
//
/***********************************************************************
Program:	System Matrix Numerical Calculation Algorithm Based on GPU
function:	System matrix numerical calculation for self-collimating SPECT system
Author:		Debin Zhang
Modify:     Liang Guo
Version:	2.0
Date:		2020-10-20
***********************************************************************/



#include "photodetector.h"
#include <fstream> 
#include <stdio.h>  
#include <stdlib.h>
#include "IniFile.h"

#include <math.h>
#include <time.h>   
#include <iostream>

#include<cuda_runtime.h>
#include <device_launch_parameters.h>


#define _USE_MATH_DEFINES
#define max(a,b) ((a>=b)?a:b) 
#define min(a,b) ((a<=b)?a:b)

FILE* fid;


using namespace std;

//choose the gpu
bool InitCUDA()
{
	int devicesCount;
	cudaGetDeviceCount(&devicesCount);

	cout << "counts = " << devicesCount << endl;

	if (devicesCount == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}

	int i;

	for (i = 0; i < devicesCount; i++) {
		cudaDeviceProp prop;
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if (prop.major >= 1) {
				break;
			}
		}
	}

	int selectIndex;

	cout << "please enter the gpu index you'd like to choose: ";
	cin >> selectIndex;
	cout << "The chosen GPU index is: " << selectIndex << endl;

	cudaSetDevice(selectIndex);

	return true;
}

int main(int argc, char *argv[])
{

	ReadInputJson();
	//parameters
	float* parameter = new float[1280];
	string pathnames[10];
	parameter = Parameters(parameter);

	string sysmatPath = GetSysmatPath();
	string FullName;
	
	int metalPlateMode = parameter[900];
	int crystalArrayMode = parameter[901];

	const int numProjectionSingle = parameter[1] * parameter[3];
	//const int numImagebin = parameter[1017] * parameter[1018] * parameter[1019];
	const int numImagebin = parameter[1017] * parameter[1018] * parameter[1000];

	const int numRotation = parameter[17];

	cout << "########################" << endl;
	cout << "numProjectionSingle = " << numProjectionSingle << endl;
	cout << "numImagebin = " << numImagebin << endl;
	cout << "########################" << endl;

	float* out = new float[numProjectionSingle * numImagebin]();

	InitCUDA();

	cout << "CUDA initialized." << endl;

	/************************************* collimation layout end *************************************************/

	int collistartidx, colliendidx, startidxDetectorZ,endidxDetectorZ,startidxImageZ,endidxImageZ;
	int numRot,startRot,endRot;
	float translationLength;
	int numTranslation;

	cout << "please enter the startidxImageZ: ";
	cin >> startidxImageZ;
	//cout << "please enter the idxImageZinterval: ";
	//cin >> idxImageZinterval;
	cout << "please enter the endidxImageZ: ";
	cin >> endidxImageZ;

	cout << "please enter the start index of idxDetectorZ:";
	cin >> startidxDetectorZ;

	cout << "please enter the end index of idxDetectorZ:";
	cin >> endidxDetectorZ;

	cout << "Please enter the total length of 2D translation in mm: ";
	cin >> translationLength;

	cout << "Please enter the total number of 2D translation steps: ";
	cin >> numTranslation;

	parameter[800] = translationLength;
	parameter[801] = numTranslation;
	
	for (int idxTranslationZ = 0; idxTranslationZ < numTranslation; idxTranslationZ++)
	{
		parameter[803] = idxTranslationZ;
			for (int idxImageZ = startidxImageZ; idxImageZ < endidxImageZ; idxImageZ = idxImageZ + (int)parameter[1000])
			{
				//int idxImageZ = 0;
				for (int idxDetectorZ = startidxDetectorZ; idxDetectorZ < endidxDetectorZ; idxDetectorZ++)
				{
					//int idxDetectorZ = 0;  

					parameter[301] = idxDetectorZ; // 0; //st  art idxDetectorZ
					parameter[302] = idxImageZ; // 0; //start idxDetectorZ
					/*float* out = new float[numProjectionSingle * numImagebin * numRotation]();*/
					for (int idxRotation = 0; idxRotation < numRotation; idxRotation++){
						delete(out);
						float * out = new float[numProjectionSingle * numImagebin]();
						cout << "########################" << endl;
						cout << "Rotation (" << idxRotation << ") processing ..." << endl;
						cout << "########################" << endl;
						parameter[18] = idxRotation;

						clock_t start1, start2, finish;
						double duration_total, duration_recon;

						start1 = clock();

						cout << "idxDetectorZ = " << idxDetectorZ << "; idxImageZ = " << idxImageZ << " starting..." << endl;

						int q = photodetector(parameter, out);

						finish = clock();
						duration_recon = (double)(finish - start1) / CLOCKS_PER_SEC;
						cout << "calculation time : " << duration_recon << endl;

						char Fname[2048];
						
						if (metalPlateMode == 1)
						{
							//sprintf(Fname, "/sysmat_%0.flayer_Rot_%d_of_%d_2mmslitin10mm_%d_idxT%d_numT%din%0.fmm_IZ%d_DZ%d_100.sysmat", parameter[2],idxRotation,numRotation, 1010+crystalArrayMode*90, idxTranslationZ, numTranslation,translationLength,idxImageZ,idxDetectorZ);
							sprintf(Fname, "sysmatMatrix.sysmat");
							//sprintf(Fname, "sysmat/sysmat_%0.flayer_Rot_%d_of_%d_2mmslitin10mm_%d_idxT%d_numT%din%0.fmm_IZ%d_DZ%d_100.sysmat", parameter[2],idxRotation,numRotation, 1010+crystalArrayMode*90, idxTranslationZ, numTranslation,translationLength,idxImageZ,idxDetectorZ);
						}
						else
						{
							//sprintf(Fname, "/sysmat_%0.flayer_Rot_%d_of_%d_noslit_%d_idxT%d_numT%din%0.fmm_IZ%d_DZ%d_100.sysmat", parameter[2],idxRotation,numRotation, 1010 + crystalArrayMode * 90, idxTranslationZ, numTranslation, translationLength, idxImageZ, idxDetectorZ);
							sprintf(Fname, "sysmatMatrix.sysmat");
							//sprintf(Fname, "sysmat/sysmat_%0.flayer_Rot_%d_of_%d_noslit_%d_idxT%d_numT%din%0.fmm_IZ%d_DZ%d_100.sysmat", parameter[2],idxRotation,numRotation, 1010 + crystalArrayMode * 90, idxTranslationZ, numTranslation, translationLength, idxImageZ, idxDetectorZ);
						}
						FILE* fp1;
						FullName = sysmatPath + string(Fname);
						char* c = const_cast<char*>(FullName.c_str());
						fp1 = fopen(c, "wb+");
						if (fp1 == 0) { puts("error"); exit(0); }
						fwrite(out, sizeof(float), numProjectionSingle * numImagebin, fp1);
						fclose(fp1);
					}
				}
			}
			//cout << "idxDetectorZ = " << idxDetectorZ << "; idxImageZ = " << idxImageZ <<" done" << endl;
	}


	delete[] parameter;
	delete[] out;

	cout << "########################" << endl;
	cout << "Sysmat written." << endl;
	cout << "########################" << endl;

	return 0;
}


