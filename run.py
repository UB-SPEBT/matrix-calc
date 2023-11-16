
#! /usr/bin/python3 

import os
import time

CUDA_VERSION_11_8_0 = True
# To use modified solid angle calculation, set this flag True
MSAC = False


# commands = [
# 	'nvcc --version',
# 	'nvcc -O2 -c photodetector.cu',
# 	'g++ -O2 -c PhotoDetector.cpp  -I/usr/local/cuda/include/'
# 	'g++ -O2 -c parameters.cpp  -I/usr/local/cuda/include/',
# 	'g++ -O2 -c IniFile.cpp  -I/usr/local/cuda/include/',
# 	'nvcc -o tester photodetector.o PhotoDetector.o parameters.o IniFile.o',
# 	'./tester'
# ] 


if CUDA_VERSION_11_8_0:
	# 11.8.0
	CUDA_PATH = '/cvmfs/soft.ccr.buffalo.edu/versions/2023.01/easybuild/software/Core/cuda/11.8.0/include'
	ADDITIONAL_PARAM = '-Xcompiler "-Xlinker -rpath=/opt/software/nvidia/lib64"'
else:
	# 11.3.1
	CUDA_PATH = '/usr/local/cuda/include/'
	ADDITIONAL_PARAM = ''

if MSAC:
	PHOTODETECTOR_CUDA = 'photodetector.cu'
else:
	PHOTODETECTOR_CUDA = "photodetector_msac.cu"

commands = [
	'nvcc --version',
	'nvcc -O2 -c '+PHOTODETECTOR_CUDA,
	'g++ -O2 -c PhotoDetector.cpp  -I'+CUDA_PATH,
	'g++ -O2 -c parameters.cpp  -I'+CUDA_PATH,
	'g++ -O2 -c IniFile.cpp  -I'+CUDA_PATH,
	'nvcc -o tester '+ADDITIONAL_PARAM+' photodetector.o PhotoDetector.o parameters.o IniFile.o',
	'./tester'
] 


for command in commands:
	# os.system(command)
	print(command)
	time.sleep(1) 

print(os.getcwd())
os.chdir('../')
print(os.getcwd())