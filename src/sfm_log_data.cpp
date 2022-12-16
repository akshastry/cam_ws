// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2019 Intel Corporation. All Rights Reserved.
#include <librealsense2/rs.hpp>

// C++ headers
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <execinfo.h>

// ROS headers
#include "ros/ros.h"
#include "geometry_msgs/Twist.h"
#include "geometry_msgs/Vector3.h"
#include "std_msgs/UInt8.h"
#include "tf/tf.h"
// #include "std_msgs/Float64.h"

// #include <csvfile.h>
#include <iostream>
#include <fstream>

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

/* Obtain a backtrace and print it to stdout. */
void print_trace (void)
{
  void *array[10];
  char **strings;
  int size, i;

  size = backtrace (array, 10);
  strings = backtrace_symbols (array, size);
  if (strings != NULL)
  {

    printf ("Obtained %d stack frames.\n", size);
    for (i = 0; i < size; i++)
      printf ("%s\n", strings[i]);
  }

  free (strings);
}

int main(int argc, char** argv) 
try
{
    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;
    // Create a configuration for configuring the pipeline with a non default profile
    rs2::config cfg;
    // Add pose stream
    cfg.enable_stream(RS2_STREAM_POSE, RS2_FORMAT_6DOF);
    // Start pipeline with chosen configuration
    pipe.start(cfg);


    double pitch, roll, yaw;

    tf::Quaternion q;
    tf::Matrix3x3 R;


    // File IO
    try{
        std::ofstream myfile;
        myfile.open ("/home/aero/Desktop/cam_ws/src/Tracker/SFM_images/SFM_data.csv", std::ofstream::out | std::ofstream::trunc);
        myfile << "Image_no, T_x, T_y, T_z, Roll, pitch, yaw\n"; // Header
        myfile.close();
    }
    catch (const std::exception &ex)
    {
        std::cout << "Exception was thrown: " << ex.what() << std::endl;
    }


    // Video stream
    VideoCapture cap(stoi(argv[1]), CAP_V4L2); 
    cap.set(CAP_PROP_FPS, 120.0); 
    cap.set(CAP_PROP_FRAME_WIDTH,1920);
    cap.set(CAP_PROP_FRAME_HEIGHT,1200);
    cap.set(CAP_PROP_GAIN, 40);

    // intrinsic camera parameters
    //      [ fx   0  cx ]
    //      [  0  fy  cy ]
    //      [  0   0   1 ]
    Mat K_matrix = Mat::zeros(3, 3, CV_64FC1); // C1 means 1 channel can be upto C4 in opencv
    K_matrix.at<double>(0, 0) = 1028.7226; //fx, std_dev +- 0.4608     
    K_matrix.at<double>(1, 1) = 1028.8745; //fy, std_dev +- 0.4441     
    K_matrix.at<double>(0, 2) = 978.0923; //cx, std_dev +- 0.2824     
    K_matrix.at<double>(1, 2) = 584.8527; //cy, std_dev +- 0.2573
    K_matrix.at<double>(2, 2) = 1;

    // radial distortion coefficients
    Mat distCoeffs = Mat::zeros(5, 1, CV_64FC1); 
    distCoeffs.at<double>(0) = -0.34052699; //k1, std_dev +- 0.00036051
    distCoeffs.at<double>(1) = +0.13226244; //k2, std_dev +- 0.00038455
    distCoeffs.at<double>(2) = -0.0; //p1, std_dev +- 0.000
    distCoeffs.at<double>(3) = +0.0; //p2, std_dev +- 0.000
    distCoeffs.at<double>(4) = -0.02500858; //k3, std_dev +- 0.00012857

    // Image ctr
    int img_no = 1;

    // // Main loop
    while (1)
    {
        // Wait for the next set of realsense frames
        auto frames = pipe.wait_for_frames();

        // instantly get current frame from video stream
        Mat image;
        cap >> image;

        // If the frame is empty, continue to the next frame
        if (image.empty())
        {
          cout <<" ERROR: empty frame received!! \n";
          continue; 
        }

        // convert to grayscale
        if(image.channels() == 3)
        {
          cvtColor(image, image, COLOR_BGR2GRAY);
        }

        //undistort frames
        Mat rectified_image;
        undistort(image, rectified_image, K_matrix, distCoeffs);

        // From the frameset that arrives get a frame of RS2_STREAM_POSE type
        auto f = frames.first_or_default(RS2_STREAM_POSE);
        // Cast the frame to pose_frame and get its data
        auto pose_data = f.as<rs2::pose_frame>().get_pose_data();


        if(pose_data.tracker_confidence >= 1)
        {

            // save video frame
            imwrite(string("/home/aero/Desktop/cam_ws/src/Tracker/SFM_images/") + to_string(img_no) + string(".png"), rectified_image);

            // get roll, pitch, yaw
            q.setW(pose_data.rotation.w);
            q.setX(pose_data.rotation.x);
            q.setY(pose_data.rotation.y);
            q.setZ(pose_data.rotation.z);
            R.setRotation(q);

            R.getEulerYPR(yaw, pitch, roll);

            // save translation and rotation
            try{
                std::ofstream myfile;
                myfile.open ("/home/su/Neo_WS/src/RealSense/SFM_images/SFM_data.csv", std::ofstream::out | std::ofstream::app);
                myfile << img_no << "," << pose_data.translation.x << ","  << pose_data.translation.y << ","  << pose_data.translation.z << ","  << roll << ","  << pitch << ","  << yaw << ","  << endl;
                myfile.close();
            }
            catch (const std::exception &ex)
            {
                std::cout << "Exception was thrown: " << ex.what() << std::endl;
            }

            // increment image counter
            img_no++;

        }

    // Press  ESC on keyboard to exit
    char c=(char)waitKey(25);
    if(c==27)
      break;
        
    }

    // When everything done, release the video capture object
    cap.release();

    // Closes all the frames
    destroyAllWindows();

    return EXIT_SUCCESS;

}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch(...)
{
    printf("An exception occurred in pose.cpp .\n");
    print_trace();
}