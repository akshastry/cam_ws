
cmake_minimum_required(VERSION 2.8.3)
project(Tracker)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

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

catkin_package(
	INCLUDE_DIRS include
	DEPENDS OpenCV
)

message(STATUS "STarting")
link_directories(${OpenCV_LIBRARY_DIRS} src/parallel_sift/)
include_directories(include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})


add_executable(capture src/test_capture.cpp)
add_executable(video_object_detect src/video_object_detect_sift_homography_rejection_undistort_pose2_tracking.cpp)
add_executable(focus src/video_focus.cpp)

target_link_libraries(capture ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})
target_link_libraries(video_object_detect ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})
target_link_libraries(focus ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})

install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)


