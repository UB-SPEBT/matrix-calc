#include <string>
#include "IniFile.h"
#ifndef _PHOTODETECTORCUDA_H_
#define _PHOTODETECTORCUDA_H_

float* Parameters(float* parameter);

int ReadParFile(string ParameterFileName);

extern int photodetector(
	float* parameter, 
	float* dst);


#endif //_PHOTODETECTORCUDA_H_