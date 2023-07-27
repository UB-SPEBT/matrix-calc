
#! /usr/bin/python3 

import os
import time


commands = ['nvcc --version','nvcc -O2 -c photodetector.cu','g++ -O2 -c PhotoDetector.cpp  -I/usr/local/cuda/include/'
		'g++ -O2 -c parameters.cpp  -I/usr/local/cuda/include/','g++ -O2 -c IniFile.cpp  -I/usr/local/cuda/include/',
		'nvcc -o tester photodetector.o PhotoDetector.o parameters.o IniFile.o','./tester'] 


for command in commands:
	os.system(command)
	time.sleep(1) 

print(os.getcwd())
os.chdir('../')
print(os.getcwd())