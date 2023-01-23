# Matrix-calc

Ray-tracing analytical calculation of the system matrix of the scanner.

## How to Run the Code

The code needs to be compiled first. The executable will be in `./bin` folder.

## How to Compile the Code

Use CMake to compile.

1. ```cmake -B build -G Ninja``` or simply ```cmake -B build```

2. ```ninja -C build``` or ```make -C build``` , if use *GNU make*

### Dependencies
These packages need to be setup on the system before you can compile the code.

+ CMake
+ C++ Compiler which support C++ 11
+ CUDA