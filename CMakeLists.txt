cmake_minimum_required(VERSION 3.20)
Project(TimeOptimalAttitudeControl)

set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

SET(CMAKE_EXE_LINKER_FLAGS 
    "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,${CMAKE_SOURCE_DIR}/build")

# for using casadi
find_package(casadi REQUIRED)
# For multi threading
find_package(Threads REQUIRED)

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

# set the Ipopt library and include directory
set(LIBRARY_DIRS /usr/local/lib)
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

if(IPOPT_LIBRARY AND IPOPT_INCLUDE_DIRS)
    set(IPOPT_LIBRARIES ${IPOPT_LIBRARIES} ${IPOPT_LIBRARY})
    message(STATUS "Found IPOPT library: ${IPOPT_LIBRARY}")
else()
    message(FATAL_ERROR "Ipopt library not found. Please set IPOPT_LIBRARY and IPOPT_INCLUDE_DIRS.")
endif()

# find the following packages
find_package(Threads REQUIRED)

add_library(toac STATIC src/lib/optimizer.cpp src/lib/dynamics.cpp src/lib/constraints.cpp src/lib/plots.cpp)
target_include_directories(
    toac PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/include/toac
)
# include header directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/toac
    SYSTEM ${IPOPT_INCLUDE_DIRS}
    SYSTEM ${CASADI_INCLUDE_DIR}
    ${CODE_GEN_INCLUDE_DIR})

# library directories
link_directories(${LIBRARY_DIRS})

# find all the header files
file(GLOB HEADER_FILES_HPP ${CMAKE_SOURCE_DIR}/include/*.hpp)
file(GLOB HEADER_FILES_H ${CMAKE_SOURCE_DIR}/include/*.h)
file(GLOB HEADER_FILES_H ${CMAKE_SOURCE_DIR}/include/toac/*.h)

add_executable(GenerateCode src/c_code_gen_sol.cpp)
# Link the executable to the CasADi library
target_link_libraries(GenerateCode PRIVATE ${CASADI_LIBRARIES} ${IPOPT_LIBRARIES} toac)
target_include_directories(GenerateCode PRIVATE ${CASADI_INCLUDE_DIRS} ${IPOPT_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/include/toac)

add_executable(SerialBangbangSymmetric src/serial/bangbang/symmetric/main.cpp)

# Link the executable to the CasADi library
target_link_libraries(SerialBangbangSymmetric PRIVATE ${CASADI_LIBRARIES} toac)
target_include_directories(SerialBangbangSymmetric PRIVATE ${CASADI_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/include/toac)

add_executable(SerialBangbangSymmetricCvodes src/serial/bangbang/symmetric/main_cvodes.cpp)

# Link the executable to the CasADi library
target_link_libraries(SerialBangbangSymmetricCvodes PRIVATE ${CASADI_LIBRARIES} toac)
target_include_directories(SerialBangbangSymmetricCvodes PRIVATE ${CASADI_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/include/toac)

# Runs GenerateCode automatically after build
add_custom_target(run_codegen ALL
    COMMAND GenerateCode
    DEPENDS GenerateCode
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running GenerateCode automatically after build"
)