cmake_minimum_required(VERSION 3.20)
Project(TimeOptimalAttitudeControl)

set(CMAKE_CXX_STANDARD 20)

target_include_directories(toac PUBLIC include)

# add_library(toac STATIC src/lib/mylibrary1.cpp src/lib/mylibrary2.cpp)

add_executable(SerialBangbangSymmetric src/seial/bangbang/symmetric/main.cpp)
target_link_libraries(SerialBangbangSymmetric PRIVATE toac)

add_executable(SerialBangbangAsymmetric src/seial/bangbang/asymmetric/main.cpp)
target_link_libraries(SerialBangbangAsymmetric PRIVATE toac)
