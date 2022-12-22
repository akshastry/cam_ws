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

    Mat invK_matrix = Mat::zeros(3, 3, CV_64FC1); // C1 means 1 channel can be upto C4 in opencv
    invK_matrix.at<double>(0, 0) = 1.0 / K_matrix.at<double>(0, 0); //1/fx
    invK_matrix.at<double>(1, 1) = 1.0 / K_matrix.at<double>(1, 1); //1/fy     
    invK_matrix.at<double>(0, 2) = -K_matrix.at<double>(0, 2) / K_matrix.at<double>(0, 0); //-cx/fx     
    invK_matrix.at<double>(1, 2) = -K_matrix.at<double>(1, 2) / K_matrix.at<double>(1, 1); //-cy/fy
    invK_matrix.at<double>(2, 2) = 1;    

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
    MATCHEDpoints1.push_back(keypoints1.at(x.queryIdx).pt);
    MATCHEDpoints2.push_back(keypoints2.at(x.trainIdx).pt);
    }


    vector<uchar> mask;
    Mat E_mat = findEssentialMat( MATCHEDpoints1, MATCHEDpoints2, K_matrix, RANSAC, 0.999, 1.0, mask);

    // draw the inlier matches in green and outlier in red
    Mat OutImg;

    vector<char> char_mask;
    for(auto x: mask)
        char_mask.push_back(x);

    drawMatches(rectified_image1, keypoints1, rectified_image2, keypoints2, matches, OutImg, Scalar(0,255,0), Scalar::all(-1),
                 char_mask, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);


    resize(OutImg, OutImg, cv::Size(), 0.5, 0.5);
    imshow("inlier matches", OutImg);

    vector<char> invert_mask;
    for(auto x: char_mask)
        invert_mask.push_back(1-(int)x);
    drawMatches(rectified_image1, keypoints1, rectified_image2, keypoints2, matches, OutImg, Scalar(0,0,255), Scalar::all(-1),
                 invert_mask, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    resize(OutImg, OutImg, cv::Size(), 0.5, 0.5);
    imshow("outlier matches", OutImg);
    waitKey(0); // Wait for a keystroke in the window

    //get inliers and outliers
    vector<Point2f> Inlier_points1, Inlier_points2, Outlier_points1, Outlier_points2;
    for (int i=0; i<mask.size(); i++)
    {
        if(mask.at(i))
        {
            Inlier_points1.push_back(MATCHEDpoints1.at(i));
            Inlier_points2.push_back(MATCHEDpoints2.at(i));
        }
        else
        {
            Outlier_points1.push_back(MATCHEDpoints1.at(i));
            Outlier_points2.push_back(MATCHEDpoints2.at(i));
        }
    }

    // decompose essential matrix
    Mat Rot1, Rot2, t;
    decomposeEssentialMat(E_mat, Rot1, Rot2, t);
    Tdisp = t;
    Rot = Rot2;

    // pi :)
    double pi = 4.0*atan(1.0);
    // R3 is rotation of camera wrt to realsense axes
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

    // T3 is location of camera wrt to realsense center in realsense coordinate axes
    Mat T3 = Mat::zeros(3,1, CV_64FC1);
    T3.at<double>(0) = 0.01;
    T3.at<double>(1) = 0.03;
    T3.at<double>(2) = 0.01;

    // rotation and translation from relasense
    Mat rot = R3.t() * R1.t() * R2 * R3;
    Mat tdisp =  (R2*R3).t() * (T1 - T2 + (R1 - R2)*T3 );
    tdisp = tdisp / norm(tdisp);


    // display rotation and translation from realsense
    cout << "translation epipolar: " << endl;
    cout << Tdisp << endl;
    cout << "normalized realsense translation: " << endl;
    cout << tdisp / norm(tdisp) << endl << endl;

    cout << "rotation epipolar: " << endl;
    cout << Rot1 << endl;
    cout << Rot2 << endl;
    cout << "rotation realsense: " << endl;
    cout << rot.t() << endl << endl;
        
    
    // display initial determined essential matrix and those composed from rotation and translation
    cout << "Initial E_mat: " << endl;
    E_mat = E_mat / norm(E_mat);
    cout << E_mat << endl << endl; 
    
    cout << "Composed E_mat: " << endl;
    Mat Emat1 = crossmat(Tdisp.at<double>(0), Tdisp.at<double>(1), Tdisp.at<double>(2)) * Rot2;
    Emat1 = Emat1 / norm(Emat1);
    cout << Emat1 << endl << endl;

    Mat Emat2 = crossmat(Tdisp.at<double>(0), Tdisp.at<double>(1), Tdisp.at<double>(2)) * Rot1;
    Emat2 = Emat2 / norm(Emat2);
    cout << Emat2 << endl << endl;

    Mat E_mat2 = crossmat(tdisp.at<double>(0), tdisp.at<double>(1), tdisp.at<double>(2)) * rot.t();
    E_mat2 = E_mat2 / norm(E_mat2);
    cout << E_mat2 << endl << endl;
    

    // diplay residuals of inliers and outliers
    vector<double> inlier_res, outlier_res;
    cout << endl << "Inlier values: " << endl;
    for(int i = 0; i < Inlier_points1.size(); i++)
    {
        Mat p2 = Mat::zeros(3,1, CV_64FC1);
        Mat p1 = Mat::zeros(3,1, CV_64FC1);

        p2.at<double>(0) = Inlier_points2.at(i).x;
        p2.at<double>(1) = Inlier_points2.at(i).y;
        p2.at<double>(2) = 1.0;

        p1.at<double>(0) = Inlier_points1.at(i).x;
        p1.at<double>(1) = Inlier_points1.at(i).y;
        p1.at<double>(2) = 1.0;

        Mat res = p2.t() * invK_matrix.t() * E_mat2 * invK_matrix * p1;
        inlier_res.push_back(res.at<double>(0));

        cout << res.at<double>(0) << ", ";
    }
    cout << endl;
    

    cout << endl << "Outlier values: " << endl;
    for(int i = 0; i < Outlier_points1.size(); i++)
    {
        Mat p2 = Mat::zeros(3,1, CV_64FC1);
        Mat p1 = Mat::zeros(3,1, CV_64FC1);

        p2.at<double>(0) = Outlier_points2.at(i).x;
        p2.at<double>(1) = Outlier_points2.at(i).y;
        p2.at<double>(2) = 1.0;

        p1.at<double>(0) = Outlier_points1.at(i).x;
        p1.at<double>(1) = Outlier_points1.at(i).y;
        p1.at<double>(2) = 1.0;

        Mat res = p2.t() * invK_matrix.t() * E_mat2 * invK_matrix * p1;
        outlier_res.push_back(res.at<double>(0));

        cout << res.at<double>(0) << ", ";
    }
    cout << endl;

    // mean and variance of inlier and outlier residuals
    double inlier_res_mean = accumulate(inlier_res.begin(), inlier_res.end(), 0.0) / inlier_res.size();
    for_each(inlier_res.begin(), inlier_res.end(), [inlier_res_mean](double &v){
        v = v - inlier_res_mean;
        v = v * v;
    });
    double inlier_res_var = accumulate(inlier_res.begin(), inlier_res.end(), 0.0) / inlier_res.size();


    double outlier_res_mean = accumulate(outlier_res.begin(), outlier_res.end(), 0.0) / outlier_res.size();
    for_each(outlier_res.begin(), outlier_res.end(), [outlier_res_mean](double &v){
        v = v - outlier_res_mean;
        v = v * v;
    });
    double outlier_res_var = accumulate(outlier_res.begin(), outlier_res.end(), 0.0) / outlier_res.size();


    cout << endl << "inlier_res_mean: " << inlier_res_mean << ", inlier_res_var: " << inlier_res_var << ", inlier_res_stddev: " << sqrt(inlier_res_var) << endl;
    cout << endl << "outlier_res_mean: " << outlier_res_mean << ", outlier_res_var: " << outlier_res_var << ", outlier_res_stddev: " << sqrt(outlier_res_var) << endl;


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