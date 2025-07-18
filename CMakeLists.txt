cmake_minimum_required(VERSION 3.20)
Project(TimeOptimalAttitudeControl)

set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

SET(CMAKE_EXE_LINKER_FLAGS 
    "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,${CMAKE_SOURCE_DIR}/build")

# for using casadi
find_package(casadi REQUIRED)
if (NOT casadi_FOUND)
    message(FATAL_ERROR "CasADi not found. Please install CasADi or set CASADI_DIR.")
else()
    message(STATUS "CasADi version: ${casadi_VERSION}")
endif()
# For multi threading
find_package(Threads REQUIRED)
# For integration
find_package(SUNDIALS REQUIRED)
if (NOT SUNDIALS_FOUND)
    message(FATAL_ERROR "SUNDIALS not found. Please install SUNDIALS or set SUNDIALS_DIR.")
else()
    message(STATUS "SUNDIALS version: ${SUNDIALS_VERSION}")
endif()
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

# SUNDIALS Configuration
set(SUNDIALS_INCLUDE_DIR /usr/local/sundials-double/include)
set(SUNDIALS_LIBRARIES_DIR /usr/local/sundials-double/lib)

# Find all required SUNDIALS libraries
find_library(SUNDIALS_CVODES_LIBRARY
    NAMES sundials_cvodes
    HINTS ${SUNDIALS_LIBRARIES_DIR} 
)

find_library(SUNDIALS_NVECCUDA_LIBRARY
    NAMES sundials_nveccuda
    HINTS ${SUNDIALS_LIBRARIES_DIR} 
)

find_library(SUNDIALS_SUNMATRIXCUSPARSE_LIBRARY
    NAMES sundials_sunmatrixcusparse
    HINTS ${SUNDIALS_LIBRARIES_DIR} /usr/local/lib /usr/lib /opt/lib
)

find_library(SUNDIALS_SUNLINSOLCUSOLVERSP_LIBRARY
    NAMES sundials_sunlinsolcusolversp
    HINTS ${SUNDIALS_LIBRARIES_DIR} /usr/local/lib /usr/lib /opt/lib
)

find_library(SUNDIALS_GENERIC_LIBRARY
    NAMES sundials_generic
    HINTS ${SUNDIALS_LIBRARIES_DIR} /usr/local/lib /usr/lib /opt/lib
)

# Check if all libraries were found
if(SUNDIALS_CVODES_LIBRARY AND SUNDIALS_NVECCUDA_LIBRARY AND 
   SUNDIALS_SUNMATRIXCUSPARSE_LIBRARY AND SUNDIALS_SUNLINSOLCUSOLVERSP_LIBRARY AND
   SUNDIALS_GENERIC_LIBRARY)
    set(SUNDIALS_LIBRARIES 
        ${SUNDIALS_CVODES_LIBRARY}
        ${SUNDIALS_NVECCUDA_LIBRARY}
        ${SUNDIALS_SUNMATRIXCUSPARSE_LIBRARY}
        ${SUNDIALS_SUNLINSOLCUSOLVERSP_LIBRARY}
        ${SUNDIALS_GENERIC_LIBRARY}
    )
    message(STATUS "All SUNDIALS libraries found:")
    message(STATUS "  CVODES: ${SUNDIALS_CVODES_LIBRARY}")
    message(STATUS "  NVECCUDA: ${SUNDIALS_NVECCUDA_LIBRARY}")
    message(STATUS "  SUNMATRIXCUSPARSE: ${SUNDIALS_SUNMATRIXCUSPARSE_LIBRARY}")
    message(STATUS "  SUNLINSOLCUSOLVERSP: ${SUNDIALS_SUNLINSOLCUSOLVERSP_LIBRARY}")
    message(STATUS "  GENERIC: ${SUNDIALS_GENERIC_LIBRARY}")
else()
    message(FATAL_ERROR "Some SUNDIALS libraries not found. Please check SUNDIALS installation.")
endif()

# Create the library
add_library(toac STATIC 
            src/lib/optimizer.cpp 
            src/lib/dynamics.cpp 
            src/lib/constraints.cpp 
            src/lib/plots.cpp 
            src/lib/CUDAdynamics.cu 
            src/lib/CustomCallback.cpp
) 

# Set CUDA properties
set_property(TARGET toac PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# CORRECT: Set include directories for toac (header paths only)
target_include_directories(toac PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/toac
    ${SUNDIALS_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# CORRECT: Link libraries to toac
target_link_libraries(toac PUBLIC
    ${CASADI_LIBRARIES}
    ${SUNDIALS_LIBRARIES}
    CUDA::cudart
    CUDA::cusparse
    CUDA::cusolver
)

# include header directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/toac
    SYSTEM ${IPOPT_INCLUDE_DIRS}
    SYSTEM ${CASADI_INCLUDE_DIR}
    SYSTEM ${SUNDIALS_INCLUDE_DIR}
    ${CODE_GEN_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# find all the header files
file(GLOB HEADER_FILES_HPP ${CMAKE_SOURCE_DIR}/include/*.hpp)
file(GLOB HEADER_FILES_H ${CMAKE_SOURCE_DIR}/include/*.h)
file(GLOB HEADER_FILES_H ${CMAKE_SOURCE_DIR}/include/toac/*.h)

add_executable(GenerateCode src/serial/bangbang/symmetric/c_code_gen.cpp)

# Link the executable to the libraries
target_link_libraries(GenerateCode PRIVATE ${CASADI_LIBRARIES} ${IPOPT_LIBRARIES} ${SUNDIALS_LIBRARIES} toac)
target_include_directories(GenerateCode PRIVATE ${CASADI_INCLUDE_DIR} ${IPOPT_INCLUDE_DIRS} ${SUNDIALS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/include/toac)

add_executable(SerialBangbangSymmetric src/serial/bangbang/symmetric/main.cpp)

# Link the executable to the libraries
target_link_libraries(SerialBangbangSymmetric PRIVATE ${CASADI_LIBRARIES} ${SUNDIALS_LIBRARIES} toac)
target_include_directories(SerialBangbangSymmetric PRIVATE ${CASADI_INCLUDE_DIR} ${SUNDIALS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/include/toac)

add_executable(SerialBangbangSymmetricCodeGen src/serial/bangbang/symmetric/main_code_gen.cpp)

# Link the executable to the libraries
target_link_libraries(SerialBangbangSymmetricCodeGen PRIVATE ${CASADI_LIBRARIES} ${SUNDIALS_LIBRARIES} toac)
target_include_directories(SerialBangbangSymmetricCodeGen PRIVATE ${CASADI_INCLUDE_DIR} ${SUNDIALS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/include/toac)

# Uncomment and fix the parallel version when needed
add_executable(ParGenerateCode src/parallel/bangbang/symmetric/c_code_gen.cpp)
target_link_libraries(ParGenerateCode PRIVATE
    ${CASADI_LIBRARIES}
    ${SUNDIALS_LIBRARIES}
    CUDA::cudart
    CUDA::cusparse
    CUDA::cusolver
    toac
)

# Runs GenerateCode automatically after build
add_custom_target(run_codegen_serial
    COMMAND GenerateCode
    DEPENDS toac GenerateCode
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running GenerateCode automatically after build"
)

add_custom_target(serial 
    COMMAND SerialBangbangSymmetricCodeGen
    DEPENDS SerialBangbangSymmetricCodeGen run_codegen_serial
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running SerialBangbangSymmetricCodeGen automatically after build"
)

add_custom_target(run_codegen_parallel
    COMMAND ParGenerateCode
    DEPENDS toac ParGenerateCode
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running ParGenerateCode automatically after build"
)

add_custom_target(parallel 
    COMMAND ParallelBangbangSymmetricCodeGen
    DEPENDS run_codegen_parallel ParallelBangbangSymmetricCodeGen
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running SerialBangbangSymmetricCodeGen automatically after build"
)