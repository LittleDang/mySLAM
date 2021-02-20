cmake_minimum_required(VERSION 2.8.3)
project(mySLAM)
set(CMAKE_BUILD_TYPE "Release")

add_compile_options(-std=c++11)

find_package(catkin REQUIRED roscpp tf)
find_package(Sophus REQUIRED  )
find_package(PCL  REQUIRED )     
find_package(DBoW3 REQUIRED )
find_package( g2o REQUIRED )

find_package(
  OpenCV 
)
include_directories(
${catkin_INCLUDE_DIRS}
${OpenCV_INCLUDE_DIR}
${Sophus_INCLUDE_DIRS}
${PCL_INCLUDE_DIRS}
${DBoW3_INCLUDE_DIRS}
${g2o_INCLUDE_DIRS}
"/usr/include/eigen3"
"include/"
)

#add_executable(cameraTest src/cameraTest.cpp)
add_executable(voTest src/voTest.cpp)
add_executable(buildGroundtruth src/buildGroundtruth.cpp)
add_executable(slamTest src/slamTest.cpp)

add_library(mySLAM_LIBS 
src/vo.cpp src/frame.cpp src/config.cpp src/loop.cpp src/viewer.cpp )
target_link_libraries(mySLAM_LIBS 
${OpenCV_LIBS} 
${catkin_LIBRARIES} 
${PCL_LIBRARIES} 
${Sophus_LIBRARIES}
${DBoW3_LIBS}
g2o_core
g2o_stuff)

#target_link_libraries(cameraTest  mySLAM_LIBS)
target_link_libraries(voTest  mySLAM_LIBS)

target_link_libraries(buildGroundtruth  mySLAM_LIBS)

target_link_libraries(slamTest  mySLAM_LIBS)
