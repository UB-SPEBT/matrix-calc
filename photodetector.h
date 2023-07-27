#include <string>
#include "IniFile.h"
#ifndef _PHOTODETECTORCUDA_H_
#define _PHOTODETECTORCUDA_H_

float* Parameters(float* parameter);

int ReadInputJson();

extern int photodetector(
	float* parameter, 
	float* dst);

string GetSysmatPath();

#endif //_PHOTODETECTORCUDA_H_