cmake_minimum_required(VERSION 3.11.0)
project(SPEBT-SYSMAT-CAL VERSION 1.0.0)
enable_language(CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Boost 1.65 REQUIRED COMPONENTS program_options)
# include(CTest)
# enable_testing()

include_directories(${PROJECT_SOURCE_DIR}/include/)
include_directories(${PROJECT_SOURCE_DIR}/include/cuda)
include_directories(${Boost_INCLUDE_DIR})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_executable(
    matcal src/main.cpp 
    )

# add_library(simplePrintf STATIC src/simplePrintf.cu)
add_library(sysMatCalc_1z_1z STATIC src/sysMatCalc_1z_1z.cu)
set_property(TARGET sysMatCalc_1z_1z PROPERTY CUDA_ARCHITECTURES "52;72") 

set_target_properties(matcal
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "lib/"
    LIBRARY_OUTPUT_DIRECTORY "lib/"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin/"
)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin)

target_link_libraries(matcal ${Boost_LIBRARIES} sysMatCalc_1z_1z)
# target_link_libraries(matcal ${Boost_LIBRARIES} )
