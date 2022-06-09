cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(fivep)

add_subdirectory(pybind11)
#find_package (Eigen3 3.3 REQUIRED)

set(CMAKE_CXX_FLAGS "-fpermissive")
find_package(OpenCV REQUIRED )

include_directories(
    ${OpenCV_LIBS}
    /home/artcs/Desktop/Keypoints/utils
)


#add_executable( five-point-nister five-point.cpp precomp.hpp usac.hpp )
pybind11_add_module( fivep five-point.cpp precomp.hpp usac.hpp)


target_link_libraries(fivep PRIVATE 
    ${OpenCV_LIBS} )
