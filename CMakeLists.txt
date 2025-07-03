cmake_minimum_required(VERSION 3.20)
Project(TimeOptimalAttitudeControl)

set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

SET(CMAKE_EXE_LINKER_FLAGS 
    "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,${CMAKE_SOURCE_DIR}/build")

# for using casadi
find_package(casadi REQUIRED)
message(STATUS "CasADi found at: ${casadi_DIR}")
# For multi threading
find_package(Threads REQUIRED)
# For integration
find_package(SUNDIALS REQUIRED)

# For GPU acceleration
find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)
message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# set the g++ compiler flags
set(CMAKE_CXX_FLAGS "-Wall -Wextra -pthread")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_LOADED)
    message(STATUS "Compiler Path: ${CMAKE_CXX_COMPILER}")
    message(STATUS "Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
    message(STATUS "Compiler Version: ${CMAKE_CXX_COMPILER_VERSION}")
endif()

# set the Ipopt include directory
set(IPOPT_INCLUDE_DIRS /usr/include/coin)

# set casadi include directory
set(CASADI_INCLUDE_DIR /usr/local/include/casadi)

# find casadi library
find_library(CASADI_LIBRARY
    NAMES casadi
    HINTS ${CASADI_INCLUDE_DIR}/../lib $ENV{CASADI_PREFIX}/lib)
if(CASADI_LIBRARY)
    set(CASADI_LIBRARIES ${CASADI_LIBRARIES} ${CASADI_LIBRARY})
endif()

# find IPOPT library
find_library(IPOPT_LIBRARY
    NAMES ipopt
    HINTS /usr/local/lib /usr/lib /opt/lib
)
if(IPOPT_LIBRARY)
    set(IPOPT_LIBRARIES ${IPOPT_LIBRARIES} ${IPOPT_LIBRARY})
else()
    message(FATAL_ERROR "Ipopt library not found. Please set IPOPT_LIBRARY.")
endif()


# find the following packages
find_package(Threads REQUIRED)

add_library(toac STATIC src/lib/optimizer.cpp src/lib/dynamics.cpp src/lib/constraints.cpp src/lib/plots.cpp) 
# add_library(toac STATIC src/lib/optimizer.cpp src/lib/dynamics.cpp src/lib/constraints.cpp src/lib/plots.cpp src/lib/CUDAdynamics.cu) 

target_include_directories(
    toac PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/include/toac
)
# include header directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/toac
    SYSTEM ${IPOPT_INCLUDE_DIRS}
    SYSTEM ${CASADI_INCLUDE_DIR}
    ${CODE_GEN_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS})

# find all the header files
file(GLOB HEADER_FILES_HPP ${CMAKE_SOURCE_DIR}/include/*.hpp)
file(GLOB HEADER_FILES_H ${CMAKE_SOURCE_DIR}/include/*.h)
file(GLOB HEADER_FILES_H ${CMAKE_SOURCE_DIR}/include/toac/*.h)

add_executable(GenerateCode src/serial/bangbang/symmetric/c_code_gen.cpp)
# add_executable(GenerateCUDACode src/parallel/bangbang/symmetric/c_code_gen.cpp)

# Link the executable to the CasADi library
target_link_libraries(GenerateCode PRIVATE ${CASADI_LIBRARIES} ${IPOPT_LIBRARIES} toac)
target_include_directories(GenerateCode PRIVATE ${CASADI_INCLUDE_DIRS} ${IPOPT_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/include/toac)

add_executable(SerialBangbangSymmetric src/serial/bangbang/symmetric/main.cpp)

# Link the executable to the CasADi library
target_link_libraries(SerialBangbangSymmetric PRIVATE ${CASADI_LIBRARIES} toac)
target_include_directories(SerialBangbangSymmetric PRIVATE ${CASADI_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/include/toac)

add_executable(SerialBangbangSymmetricCodeGen src/serial/bangbang/symmetric/main_code_gen.cpp)

# Link the executable to the CasADi library
target_link_libraries(SerialBangbangSymmetricCodeGen PRIVATE ${CASADI_LIBRARIES} toac)
target_include_directories(SerialBangbangSymmetricCodeGen PRIVATE ${CASADI_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/include/toac)

# add_executable(ParallelBangbangSymmetric src/parallel/bangbang/symmetric/main.cpp)
# target_link_libraries(ParallelBangbangSymmetric PRIVATE
#     ${CASADI_LIBRARIES}
#     SUNDIALS::cvode
#     SUNDIALS::nveccuda  
#     CUDA::cudart
#     CUDA::cusparse
#     toac
# )

# set_property(TARGET ParallelBangbangSymmetric PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Runs GenerateCode automatically after build
add_custom_target(run_codegen
    COMMAND GenerateCode
    DEPENDS GenerateCode
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running GenerateCode automatically after build"
)