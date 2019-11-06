cmake_minimum_required(VERSION 2.8)
project(l1homotopy_cpp)

## Set build type to Release by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif(NOT CMAKE_BUILD_TYPE)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
else(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
endif(CMAKE_BUILD_TYPE STREQUAL "Release")
message(STATUS "Flags: " ${CMAKE_CXX_FLAGS})

# -march=native
set(CMAKE_CXX_FLAGS "-std=c++11 -fopenmp -Wall -Wno-deprecated -Wno-ignored-qualifiers -Wno-reorder -Wno-missing-braces")

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set(BUILD_SHARED_LIBS "ON" CACHE BOOL "Build libraries as shared?")

# required libs
find_package(Eigen3 3.1.0 REQUIRED)


message(STATUS "C Flags: " ${CMAKE_C_FLAGS})
message(STATUS "CXX Flags: " ${CMAKE_CXX_FLAGS})

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

# Set source files 
list(APPEND SOURCEFILES
    src/DS_homotopy.cpp
    src/BPDN_homotopy.cpp
)
add_library( l1homotopy_cpp SHARED ${SOURCEFILES} )
target_link_libraries( l1homotopy_cpp ${EIGEN3_LIBS} )

add_executable(test1 test/test_MATLAB.cpp)
target_link_libraries( test1 l1homotopy_cpp )

