
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(sphorb)

add_subdirectory(pybind11)
#find_package (Eigen3 3.3 REQUIRED)

set(CMAKE_CXX_FLAGS "-fpermissive")
find_package(OpenCV )

include_directories(
    ${OpenCV_LIBS}
    /home/artcs/Desktop/Keypoints/SPHORB-master/include
)

#add_executable (example1 example1.cpp pfm.cpp utility.cpp detector.cpp nonmax.cpp SPHORB.cpp)
#target_link_libraries (example1 ${OpenCV_LIBS})
#add_dependencies(example1 ${PROJECT_NAME})

pybind11_add_module(sphorb_cpp example1.cpp pfm.cpp utility.cpp detector.cpp nonmax.cpp SPHORB.cpp)
target_link_libraries(sphorb_cpp PRIVATE ${OpenCV_LIBS})
#add_dependencies(sphorb_cpp ${PROJECT_NAME})
