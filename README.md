# Matrix-calc

Ray-tracing analytical calculation of the system response matrix of the scanner.

System response matrix $A$ has element on $i_{\textit{th}}$ row and $j_{\textit{th}}$ column as:

$$a_{i,j}= \frac{1}{N_j}\sum_{i' \in i}\sum_{j' \in j}\frac{\Omega_{i',j'}}{4\pi} \cdot e^{-\int_{L_s}^{L_p} \mu(l)dl}\left(1-e^{-\int_{L_p}^{L_q} \mu(l)dl}\right)$$

+ $N_j$ is the number of sub-elements in $j_{\textit{th}}$ source voxel.
+ $\Omega_{i',j'}$ s the solid angle of $i'_{\textit{th}}$ sub-element in the detector to $j'_{\textit{th}}$ sub-element in the image volumn(or FOV).
+ $L_s$ is the position of source $j'_{\textit{th}}$ sub-voxel on line L.
+ $L_p$ is the position of the photon incident on the $i'_{\textit{th}}$ sub-element in the detector.
+ $L_q$ is the position of the photon exit from $i'_{\textit{th}}$ sub-element in the detector.
+ $\mu(l)$ is the linear attenuation coefficient.

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