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

Mat crossmat(double x, double y, double z);

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

    Mat R1 = Mat::zeros(3, 3, CV_64FC1), R2 = Mat::zeros(3, 3, CV_64FC1), T1 = Mat::zeros(3, 1, CV_64FC1), T2 = Mat::zeros(3, 1, CV_64FC1); // for realsense
    Mat Rot = Mat::zeros(3, 3, CV_64FC1), Tdisp = Mat::zeros(3, 1, CV_64FC1); // foor epipolar geometry
    
    Mat image1, image2, rectified_image1, rectified_image2;


    while(1)
    {
        cap >> image1;
        imshow("image1", image1);
        // Press  ESC on keyboard to exit
        char c=(char)waitKey(25);
        if(c==27)
        {   
            destroyWindow("image1");
            break;
        }
    }


    // string s;
    // cout << "first image?: ";
    // cin >> s;

    // Wait for the next set of realsense frames
    auto frames = pipe.wait_for_frames();
    // From the frameset that arrives get a frame of RS2_STREAM_POSE type
    auto f = frames.first_or_default(RS2_STREAM_POSE);
    // Cast the frame to pose_frame and get its data
    auto pose_data = f.as<rs2::pose_frame>().get_pose_data();


    if(pose_data.tracker_confidence >= 1)
    {

        // instantly get current frame from video stream
        cap >> image1;

        // If the frame is empty, continue to the next frame
        if (image1.empty())
        {
          cerr <<" ERROR: empty frame received!! \n"; 
        }

        // convert to grayscale
        if(image1.channels() == 3)
        {
          cvtColor(image1, image1, COLOR_BGR2GRAY);
        }

        //undistort frames
        undistort(image1, rectified_image1, K_matrix, distCoeffs);

        T1.at<double>(0) = pose_data.translation.x;
        T1.at<double>(1) = pose_data.translation.y;
        T1.at<double>(2) = pose_data.translation.z;

        // get roll, pitch, yaw
        q.setW(pose_data.rotation.w);
        q.setX(pose_data.rotation.x);
        q.setY(pose_data.rotation.y);
        q.setZ(pose_data.rotation.z);
        R.setRotation(q);

        R1.at<double>(0,0) = R[0][0];
        R1.at<double>(0,1) = R[0][1];
        R1.at<double>(0,2) = R[0][2];

        R1.at<double>(1,0) = R[1][0];
        R1.at<double>(1,1) = R[1][1];
        R1.at<double>(1,2) = R[1][2];

        R1.at<double>(2,0) = R[2][0];
        R1.at<double>(2,1) = R[2][1];
        R1.at<double>(2,2) = R[2][2];

        // R.getEulerYPR(yaw, pitch, roll);
        // R1 = eul2rotmat(roll, pitch, yaw);
    }

    cout << "T1: " << endl << T1 << endl;
    cout << "R1: " << endl << R1 << endl;


    while(1)
    {
        cap >> image2;
        imshow("image2", image2);
        // Press  ESC on keyboard to exit
        char c=(char)waitKey(25);
        if(c==27)
        {   
            destroyWindow("image2");
            break;
        }
    }

    // imshow("image1", image1);

    // cout << "next image?: ";
    // cin >> s;

    // Wait for the next set of realsense frames
    frames = pipe.wait_for_frames();
    // From the frameset that arrives get a frame of RS2_STREAM_POSE type
    f = frames.first_or_default(RS2_STREAM_POSE);
    // Cast the frame to pose_frame and get its data
    pose_data = f.as<rs2::pose_frame>().get_pose_data();


    if(pose_data.tracker_confidence >= 1)
    {

        // instantly get current frame from video stream
        cap >> image2;

        // If the frame is empty, continue to the next frame
        if (image2.empty())
        {
          cerr <<" ERROR: empty frame received!! \n"; 
        }

        // convert to grayscale
        if(image2.channels() == 3)
        {
          cvtColor(image2, image2, COLOR_BGR2GRAY);
        }

        //undistort frames
        undistort(image2, rectified_image2, K_matrix, distCoeffs);

        T2.at<double>(0) = pose_data.translation.x;
        T2.at<double>(1) = pose_data.translation.y;
        T2.at<double>(2) = pose_data.translation.z;

        // get roll, pitch, yaw
        q.setW(pose_data.rotation.w);
        q.setX(pose_data.rotation.x);
        q.setY(pose_data.rotation.y);
        q.setZ(pose_data.rotation.z);
        R.setRotation(q);

        R2.at<double>(0,0) = R[0][0];
        R2.at<double>(0,1) = R[0][1];
        R2.at<double>(0,2) = R[0][2];

        R2.at<double>(1,0) = R[1][0];
        R2.at<double>(1,1) = R[1][1];
        R2.at<double>(1,2) = R[1][2];

        R2.at<double>(2,0) = R[2][0];
        R2.at<double>(2,1) = R[2][1];
        R2.at<double>(2,2) = R[2][2];

        // R.getEulerYPR(yaw, pitch, roll);
        // R2 = eul2rotmat(roll, pitch, yaw);
    }

    cout << "T2: " << endl << T2 << endl;
    cout << "unit T2: " << endl << T2 / norm(T2) << endl;
    cout << "R2: " << endl << R2 << endl << endl;

    // imshow("image2", image2);
    // cout << "Proceed: ";
    // cin >> s;

    // create SIFT detector
    Ptr<SIFT> detector = SIFT::create( );

    // get SIFT keypoints and descriptors of the two images
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    detector->detectAndCompute( rectified_image1, Mat(), keypoints1, descriptors1 );
    detector->detectAndCompute( rectified_image2, Mat(), keypoints2, descriptors2 );

    // create best-first matcher based on L2 norm
    BFMatcher matcher(NORM_L2, true);

    // match the keypoints
    vector<DMatch> matches;
    matcher.match( descriptors1, descriptors2, matches);

    cout << "No. of matches = " << matches.size() << endl << endl;

    //get the matched points and find essential matrix
    vector<Point2f> MATCHEDpoints1, MATCHEDpoints2;
    for (auto x : matches)
    {
    MATCHEDpoints1.push_back(keypoints1[x.queryIdx].pt);
    MATCHEDpoints2.push_back(keypoints2[x.trainIdx].pt);
    }

    auto E_mat = findEssentialMat( MATCHEDpoints1, MATCHEDpoints2, K_matrix, RANSAC);
    Mat Rot1, Rot2, t;
    decomposeEssentialMat(E_mat, Rot1, Rot2, t);
    Tdisp = t;
    Rot = Rot2;

    double pi = 4.0*atan(1.0);
    // R.setEulerYPR(-pi/2, 0.0, pi);
    Mat R3 = Mat::zeros(3, 3, CV_64FC1);

    R3.at<double>(0,0) = 0.0;
    R3.at<double>(1,0) = -1.0;
    R3.at<double>(2,0) = 0.0;

    R3.at<double>(0,1) = -1.0;
    R3.at<double>(1,1) = 0.0;
    R3.at<double>(2,1) = 0.0;

    R3.at<double>(0,2) = 0.0;
    R3.at<double>(1,2) = 0.0;
    R3.at<double>(2,2) = -1.0;

    // location of camera wrt to realsense center in realsense coordinate axes
    Mat T3 = Mat::zeros(3,1, CV_64FC1);
    T3.at<double>(0) = 0.01;
    T3.at<double>(1) = 0.03;
    T3.at<double>(2) = 0.01;

    Mat rot = R3.t() * R1.t() * R2 * R3;
    Mat tdisp =  (R1*R3).t() * (T2 - T1 + (R2 - R1)*T3 );

    cout << "translation epipolar: " << endl;
    cout << Tdisp << endl;
    // cout << "translation realsense: " << endl;
    // cout << tdisp << endl;
    cout << "normalized realsense translation: " << endl;
    cout << tdisp / norm(tdisp) << endl;

    // tdisp =  R1.t() * (T2 - T1);
    // cout << tdisp / norm(tdisp) << endl;
    // cout << R3.t() * (tdisp / norm(tdisp)) << endl;    

    cout << "rotation epipolar: " << endl;
    cout << Rot1 << endl;
    cout << Rot2 << endl;
    cout << "rotation realsense: " << endl;
    cout << rot.t() << endl;
        
    cout << endl << endl;
    cout << "Initial E_mat: " << endl;
    cout << E_mat << endl << endl;
    cout << "Composed E_mat: " << endl;
    // cout << Rot1 * crossmat(Tdisp.at<double>(0), Tdisp.at<double>(1), Tdisp.at<double>(2))  << endl << endl;

    // cout << Rot1.t() * crossmat(Tdisp.at<double>(0), Tdisp.at<double>(1), Tdisp.at<double>(2))  << endl << endl;

    cout << crossmat(Tdisp.at<double>(0), Tdisp.at<double>(1), Tdisp.at<double>(2)) * Rot1 << endl << endl;

    // cout << crossmat(Tdisp.at<double>(0), Tdisp.at<double>(1), Tdisp.at<double>(2)) * Rot1.t()  << endl << endl;

    tdisp = tdisp / norm(tdisp);
    cout << crossmat(tdisp.at<double>(0), tdisp.at<double>(1), tdisp.at<double>(2)) * rot.t() << endl << endl;    
    
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

Mat crossmat(double x, double y, double z)
{
    Mat ret = Mat(3, 3, CV_64FC1);

    ret.at<double>(0,0) = 0;
    ret.at<double>(1,1) = 0;
    ret.at<double>(2,2) = 0;

    ret.at<double>(0,1) = -z;
    ret.at<double>(0,2) = y;
    ret.at<double>(1,2) = -x;

    ret.at<double>(1,0) = z;
    ret.at<double>(2,0) = -y;
    ret.at<double>(2,1) = x;

    return ret;
}