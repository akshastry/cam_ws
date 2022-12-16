
cmake_minimum_required(VERSION 2.8.3)
project(Tracker)
# check c++14 / c++0x
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX14)
    set(CMAKE_CXX_FLAGS "-std=c++14")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "-std=c++0x")
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
endif()


find_package(catkin REQUIRED COMPONENTS geometry_msgs message_generation roscpp)

set(_opencv_version 4)
find_package(OpenCV 4 QUIET)
if(NOT OpenCV_FOUND)
  message(STATUS "Did not find OpenCV 4, trying OpenCV 3")
  set(_opencv_version 3)
endif()

find_package(OpenCV ${_opencv_version} REQUIRED
  COMPONENTS
    opencv_videoio
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    opencv_xfeatures2d
  CONFIG
)

find_package(Armadillo REQUIRED)

catkin_package(
	INCLUDE_DIRS include
	DEPENDS OpenCV
)

message(STATUS "Starting")
link_directories(${OpenCV_LIBRARY_DIRS} src/parallel_sift/)
include_directories(include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${ARMADILLO_INCLUDE_DIRS})


add_executable(capture src/test_capture.cpp)
add_executable(video_object_detect src/video_object_detect_sift_homography_rejection_undistort_pose2_tracking_newcam_acceleration_tuning_filter_correlation.cpp)
add_executable(focus src/video_focus.cpp)
add_executable(arma src/arma_test.cpp)
add_executable(sample src/video_object_detect_practice.cpp)
add_executable(sfm src/sfm_log_data.cpp)
add_executable(sfm2 src/SFM.cpp)
add_executable(sfm3 src/sfm_test.cpp)

target_link_libraries(capture ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})
target_link_libraries(video_object_detect ${OpenCV_LIBRARIES} ${catkin_LIBRARIES} ${ARMADILLO_LIBRARIES})
target_link_libraries(focus ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})
target_link_libraries(arma ${OpenCV_LIBRARIES} ${catkin_LIBRARIES} ${ARMADILLO_LIBRARIES})
target_link_libraries(sample ${OpenCV_LIBRARIES} ${catkin_LIBRARIES} ${ARMADILLO_LIBRARIES})
target_link_libraries(sfm librealsense2.so ${OpenCV_LIBRARIES} ${catkin_LIBRARIES} ${ARMADILLO_LIBRARIES})
target_link_libraries(sfm2 librealsense2.so ${OpenCV_LIBRARIES} ${catkin_LIBRARIES} ${ARMADILLO_LIBRARIES})
target_link_libraries(sfm3 librealsense2.so ${OpenCV_LIBRARIES} ${catkin_LIBRARIES} ${ARMADILLO_LIBRARIES})


install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)


