cmake_minimum_required(VERSION 3.20)
Project(TimeOptimalAttitudeControl)

set(CMAKE_CXX_STANDARD 20)

#include_directories(${CMAKE_SOURCE_DIR}/include)

add_library(toac STATIC src/lib/optimizer.cpp)

# add_library(toac STATIC src/lib/optimizer.cpp src/lib/mylibrary2.cpp)

add_executable(SerialBangbangSymmetric src/serial/bangbang/symmetric/main.cpp)
target_link_libraries(SerialBangbangSymmetric PRIVATE toac)

#add_executable(SerialBangbangAsymmetric src/serial/bangbang/asymmetric/main.cpp)
#target_link_libraries(SerialBangbangAsymmetric PRIVATE toac)

target_include_directories(toac PUBLIC ${CMAKE_SOURCE_DIR}/include)
