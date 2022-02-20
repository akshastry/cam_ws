#include "opencv2/opencv.hpp"
#include <iostream>
#include <stdio.h>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"

#include "ros/ros.h"
#include "geometry_msgs/Pose.h"

#include <chrono>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

#define measure_exec_time

void readme();
void Kalman_Filter(Mat &, Mat &, const double, const double, const double, const double, const double, const double);
void Kalman_predict(Mat &, Mat &, const double, const double, const double, const double);

int main( int argc, char** argv ){

  if (argc != 3 )
  { readme(); return -1; }

  // image of model
  Mat img_object = imread( String("/home/aero/Desktop/cam_ws/src/Tracker/images/")+ String(argv[2]) + String(".png"), IMREAD_GRAYSCALE );
  if( !img_object.data)
  { cout<< " --(!) Error reading model image " << endl; return -1; }

  // display model image dimensions
  int img_width = img_object.size().width;
  int img_height = img_object.size().height;

  resize(img_object, img_object, Size(img_width/5, img_height/5), INTER_AREA);
  
  // container for scene image
  Mat img_scene;

  // display model image dimensions
  img_width = img_object.size().width;
  img_height = img_object.size().height;
  
  cout << "object_image WIDTH: " << img_width << "\n";
  cout << "object_image HEIGHT: " << img_height << "\n";

  // model image physical dimensions, pose is returned in these dimensions
  float img_width_length = 24;//24.5;//25.8; // cm
  double img_height_length = 17;//17.4;//14.6; // cm

  // 3D image corners for PnP
  vector<Point3f> obj_corners3D(4);
  obj_corners3D[0] = Point3f(0, 0, 0); obj_corners3D[1] = Point3f( 0, img_width_length, 0);
  obj_corners3D[2] = Point3f(img_height_length, img_width_length, 0); obj_corners3D[3] = Point3f( img_height_length, 0, 0);

  // Initialize surf detector and extract features from the model image, containers for scene image features also created
  Ptr<SIFT> detector = SIFT::create( );
  std::vector<KeyPoint> keypoints_object, keypoints_scene;
  Mat descriptors_object, descriptors_scene;
  detector->detectAndCompute( img_object, Mat(), keypoints_object, descriptors_object );
  cout << "No. of object image features: " << keypoints_object.size() << "\n";

  // model image feature locations (used for comparing feature locations of scene image)
  vector<Point2f> obj_feature_loc;
  for (int i = 0; i<keypoints_object.size(); i++)
  {
    obj_feature_loc.push_back(keypoints_object[i].pt);
  }


  // window for displaying results
  namedWindow("Good Matches & Object detection", WINDOW_NORMAL);
  

  // Video stream
  VideoCapture cap(stoi(argv[1])); 
  cap.set(CAP_PROP_FPS, 60.0); 
  cap.set(CAP_PROP_FRAME_WIDTH,1920);
  cap.set(CAP_PROP_FRAME_HEIGHT,1200);
  // cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G'));
  img_width = cap.get(CAP_PROP_FRAME_WIDTH);
  img_height = cap.get(CAP_PROP_FRAME_HEIGHT);

  cout << "WIDTH: " << img_width << "\n";
  cout << "HEIGHT: " << img_height << "\n";

  // intrinsic camera parameters
  //      [ fx   0  cx ]
  //      [  0  fy  cy ]
  //      [  0   0   1 ]
  Mat K_matrix = Mat::zeros(3, 3, CV_64FC1);
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

  // RANSAC parameters
  float reprojectionError = 1.0;    // maximum allowed distance to consider it an inlier.

  // for solvePnP
  Mat rvec = Mat::zeros(3, 1, CV_64FC1);          // output rotation vector
  Mat tvec = Mat::zeros(3, 1, CV_64FC1);          // output translation vector
  bool useExtrinsicGuess = false;

  // 3D axes for drawing pose in the window
  vector<Point3f> axes;
  axes.push_back(Point3f(0.0, 0.0, 0.0)); // origin
  axes.push_back(Point3f(1.0, 0.0, 0.0)); // X
  axes.push_back(Point3f(0.0, 1.0, 0.0)); // Y
  axes.push_back(Point3f(0.0, 0.0, 1.0)); // Z
   
  // Check if camera opened successfully
  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }

  // kalman filter
  Mat x_k = Mat::zeros(2, 1, CV_64FC1), y_k = Mat::zeros(2, 1, CV_64FC1), z_k = Mat::zeros(2, 1, CV_64FC1);
  Mat Px_k = Mat::eye(2, 2, CV_64FC1) * 100.0, Py_k = Mat::eye(2, 2, CV_64FC1) * 100.0, Pz_k = Mat::eye(2, 2, CV_64FC1) * 100.0;
  double dt = 0.0, sigma_P2 = pow(10.0, -3.0), sigma_u2 = pow(10.0, -3.0), sigma_M2 = 0.1*pow(10.0, -5.0);
  auto KF_call_time = std::chrono::high_resolution_clock::now();

  // ros
  ros::init(argc, argv, "pose_estimator");
  ros::NodeHandle n;
  ros::Publisher pose_pub = n.advertise<geometry_msgs::Pose>("camera_pose",1000);
  geometry_msgs::Pose camera_pose;
  
  // tracking
  bool tracking = false;
  bool detected = false;
  int crop_fact = 1;
  Point2f kp_bias = Point(0,0);

  // main loop, loop over images from video until esc is pressed
  while(1){

    // to measure main loop exection time
    #if defined measure_exec_time
      auto start = std::chrono::high_resolution_clock::now();
    #endif

    // clear vectors
    keypoints_scene.clear();

    // get current frame from video
    Mat frame;
    cap >> frame;
 
    // If the frame is empty, continue to the next frame
    if (frame.empty())
    {
      cout <<" ERROR: empty frame received!! \n";
      continue; 
    }

    // convert to grayscale
    if(frame.channels() == 3)
    {
      cvtColor(frame, frame, COLOR_BGR2GRAY);
    }



    //undistort frames
    Mat rectified_frame;
    undistort(frame, rectified_frame, K_matrix, distCoeffs);
    
    //resize to a smaller frame for speed
    int scale = 2; // scale down factor
    Mat final_frame;
    resize(rectified_frame, img_scene, Size(img_width/scale, img_height/scale), INTER_AREA);
    Mat img_matches = img_scene;

    if(tracking == true)
    {
      if(detected == false)
        crop_fact++;
      else
        crop_fact = 1;

      vector<Point2f> corners(4);;
      projectPoints(obj_corners3D, rvec, tvec, K_matrix/scale, Mat::zeros(5, 1, CV_64FC1), corners);
      
      float xmin = corners[0].x, xmax = corners[0].x, ymin = corners[0].y, ymax = corners[0].y;
      for (int i=1; i<4; i++)
      {
        xmin = min(xmin, corners[i].x);
        xmax = max(xmax, corners[i].x);
        
        ymin = min(ymin, corners[i].y);
        ymax = max(ymax, corners[i].y);
      }

      Point2f pt1 = Point2f(max(0, (int)(xmin - (xmax - xmin)/2.0 * crop_fact)), max(0, (int)(ymin - (ymax - ymin)/2.0 * crop_fact)) );
      Point2f pt2 = Point2f(min(img_width/scale, (int)(xmax + (xmax - xmin)/2.0 * crop_fact)), min(img_height/scale, (int)(ymax + (ymax - ymin)/2.0 * crop_fact)) );
      Rect roi = Rect(pt1, pt2);

      if((pt1.x<=0 && pt1.y<=0 && pt2.x>=img_width && pt2.y>=img_height) || crop_fact >=10)
      {
        kp_bias.x = 0; kp_bias.y= 0;
        tracking = false;
      }
      else
      {
        kp_bias = pt1;
        // cout << "\n" << roi << "\n";
        img_scene = img_scene(roi);
      }

    }
    else
    {
      kp_bias.x = 0;
      kp_bias.y = 0;
    }



    //Detect the keypoints and extract descriptors
    detector->detectAndCompute( img_scene, Mat(), keypoints_scene, descriptors_scene );
    for (int i=0; i<keypoints_scene.size(); i++)
    {
      keypoints_scene[i].pt.x += kp_bias.x;
      keypoints_scene[i].pt.y += kp_bias.y;
    }

    // Matching descriptor vectors using FLANN or BruteForce matcher
    // FlannBasedMatcher matcher;
    BFMatcher matcher(NORM_L2);
    
    std::vector<std::vector<cv::DMatch>> matches12, matches21;
    matcher.knnMatch( descriptors_object, descriptors_scene, matches12,2 );
    matcher.knnMatch( descriptors_scene, descriptors_object, matches21,2 );

    //-- Quick calculation of min distances between keypoints
    double min_dist12 = 100;
    for( int i = 0; i < matches12.size(); i++ )
    { 
      if( matches12[i].size()>=1)
      { 
        if( matches12[i][0].distance < min_dist12 ) min_dist12 = matches12[i][0].distance;
      }
    }

    double min_dist21 = 100;
    for( int i = 0; i < matches21.size(); i++ )
    { 
      if( matches21[i].size()>=1)
      { 
        if( matches21[i][0].distance < min_dist21 ) min_dist21 = matches21[i][0].distance;
      }
    }

    // printf("-- Min dist : %f \n", min_dist12 );

    //-- Lowe's Ratio test and minimum distance threshold test 
    std::vector< DMatch > good_matches12, good_matches21;
    for( int i = 0; i < matches12.size(); i++ )
    { 
      if( matches12[i].size()>=2 && matches12[i][0].distance <= 3*min_dist12 && matches12[i][0].distance < matches12[i][1].distance * 0.7)
      { good_matches12.push_back( matches12[i][0]); }
    }

    for( int i = 0; i < matches21.size(); i++ )
    { 
      if( matches21[i].size()>=2 && matches21[i][0].distance <= 3*min_dist21 && matches21[i][0].distance < matches21[i][1].distance * 0.7)
      { good_matches21.push_back( matches21[i][0]); }
    }

    // Symmetry test
    std::vector< DMatch > best_matches;
    for (int i = 0; i<good_matches12.size(); i++)
    {
      for (int j = 0; j<good_matches21.size(); j++)
      {
        if(good_matches12[i].queryIdx == good_matches21[j].trainIdx && good_matches12[i].trainIdx == good_matches21[j].queryIdx)
        {
          best_matches.push_back(good_matches12[i]);
          break;
        }
      }
    }


    // some variables
    detected = false;
    vector< DMatch > filtered_best_matches;// container for final filtered matches    
    
    // some other checks based on relative location of features and convexity test
    if(best_matches.size() >= 4)// no. of matches to use for pose estimation
    {
      //-- Get the keypoints from the good matches
      vector<Point2f> obj;
      vector<Point2f> scene;
      for( size_t i = 0; i < best_matches.size(); i++ )
      {
        obj.push_back( keypoints_object[ best_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_scene[ best_matches[i].trainIdx ].pt);
      }

      // get homography
      Mat mask;
      Mat H = findHomography( obj, scene, RANSAC, reprojectionError, mask );

      filtered_best_matches = best_matches;

      // // check for valid homography
      // if (! H.empty())
      // { 
      //   vector<Point2f> scene_feature_loc;
      //   perspectiveTransform(obj_feature_loc, scene_feature_loc, H);

      //   // filter inliers and also based on relative location of features, (relative location one is useless probably since homography reprojection error already check that)
      //   filtered_best_matches.clear();
      //   for ( int i =0; i<best_matches.size(); i++)
      //   {
      //     Point2f diff = scene_feature_loc[best_matches[i].queryIdx] - keypoints_scene[ best_matches[i].trainIdx ].pt;
      //     if((unsigned int)mask.at<uchar>(i))// && sqrt(diff.x*diff.x + diff.y*diff.y) < reprojectionError)
      //     {
      //       filtered_best_matches.push_back(best_matches[i]);
      //     }
      //   }
      //   best_matches = filtered_best_matches;

      //   // homography from the filtered out matches
      //   if(best_matches.size() >= 4)// no. of matches to use for pose estimation
      //   {
      //     //-- Get the keypoints from the good matches
      //     obj.clear();
      //     scene.clear();
      //     for( size_t i = 0; i < filtered_best_matches.size(); i++ )
      //     {
      //       obj.push_back( keypoints_object[ filtered_best_matches[i].queryIdx ].pt );
      //       scene.push_back( keypoints_scene[ filtered_best_matches[i].trainIdx ].pt );
      //     }
      //     // do homography again but with a lower reprojection error
      //     H = findHomography( obj, scene, RANSAC, reprojectionError/2.0, mask);

          // homography should be valid
          if(!H.empty())
          {

            //Finally get the corners from the object image for perspective transformation
            vector<Point2f> obj_corners(4);
            obj_corners[0] = Point(0,0); obj_corners[1] = Point( img_object.cols, 0 );
            obj_corners[2] = Point( img_object.cols, img_object.rows ); obj_corners[3] = Point( 0, img_object.rows );
            vector<Point2f> scene_corners(4);

            // transform image corners with homography
            perspectiveTransform( obj_corners, scene_corners, H);

            // the transformed quadrilateral should be convex, else something went wrong
            if(isContourConvex(scene_corners))
            {
              detected = true;
              tracking = true;
              
              // draw matches
              drawMatches( img_object, keypoints_object, img_matches.clone(), keypoints_scene,
                 filtered_best_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                 std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

              //-- Draw lines between the corners (the mapped object in the scene - image_2 )
              line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
              line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
              line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
              line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );

              // Pnp and refinement
              solvePnP(obj_corners3D, scene_corners, K_matrix/scale, Mat::zeros(5, 1, CV_64FC1), rvec, tvec, useExtrinsicGuess, SOLVEPNP_IPPE);              

              solvePnPRefineLM(obj_corners3D, scene_corners, K_matrix/scale, Mat::zeros(5, 1, CV_64FC1), rvec, tvec,
                      TermCriteria(TermCriteria::EPS+TermCriteria::COUNT, 20, FLT_EPSILON) );

              //filter
              auto now = std::chrono::high_resolution_clock::now();
              dt = chrono::duration_cast<chrono::microseconds>(now - KF_call_time).count()/1000000.0f;
              KF_call_time = now;

              Kalman_Filter(x_k, Px_k, 0.0, tvec.at<double>(0), sigma_P2, sigma_u2, sigma_M2, dt);
              tvec.at<double>(0) = x_k.at<double>(0);

              Kalman_Filter(y_k, Py_k, 0.0, tvec.at<double>(1), sigma_P2, sigma_u2, sigma_M2, dt);
              tvec.at<double>(1) = y_k.at<double>(0);

              Kalman_Filter(z_k, Pz_k, 0.0, tvec.at<double>(2), sigma_P2, sigma_u2, sigma_M2, dt);
              tvec.at<double>(2) = z_k.at<double>(0);
              

              //publish to ros message
              camera_pose.position.x = tvec.at<double>(0);
              camera_pose.position.y = tvec.at<double>(1);
              camera_pose.position.z = tvec.at<double>(2);

              float theta = sqrt(pow(rvec.at<double>(0),2.0) + pow(rvec.at<double>(1),2.0) + pow(rvec.at<double>(2),2.0));
              camera_pose.orientation.x = rvec.at<double>(0)/theta * sin(theta/2.0);
              camera_pose.orientation.y = rvec.at<double>(1)/theta * sin(theta/2.0);
              camera_pose.orientation.z = rvec.at<double>(2)/theta * sin(theta/2.0);
              camera_pose.orientation.w = cos(theta/2.0);

              pose_pub.publish(camera_pose);


              // project axes on scene image for drawing
              vector<Point2f> axes2D;
              projectPoints(axes, rvec, tvec, K_matrix/scale, Mat::zeros(5, 1, CV_64FC1), axes2D);

              // as a sanity check for PnP, project 3D corners onto scene image, this should be same as the 2D scene corners from homography 
              scene_corners.clear();
              projectPoints(obj_corners3D, rvec, tvec, K_matrix/scale, Mat::zeros(5, 1, CV_64FC1), scene_corners);


              //-- Draw lines between the corners (the mapped object in the scene - image_2 )
              line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(255, 0, 0), 4 );
              line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar(255, 0, 0), 4 );
              line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar(255, 0, 0), 4 );
              line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar(255, 0, 0), 4 );

              // Draw the 3D axes
              line( img_matches, axes2D[0] + Point2f( img_object.cols, 0), axes2D[1] + Point2f( img_object.cols, 0), Scalar(255, 0, 0), 4 );
              line( img_matches, axes2D[0] + Point2f( img_object.cols, 0), axes2D[2] + Point2f( img_object.cols, 0), Scalar( 0, 0, 255), 4 );
              line( img_matches, axes2D[0] + Point2f( img_object.cols, 0), axes2D[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
              
              

            }
          }
      //   }
      // }
    }
    else{
      // printf("No enough matches: %ld\n", good_matches.size());
    }

    if(detected == false){
      drawMatches( img_object, keypoints_object, img_matches.clone(), keypoints_scene,
                 filtered_best_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                 std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    }
    //-- Show detected matches
    imshow( "Good Matches & Object detection", img_matches );

    #if defined measure_exec_time
    // execution time
      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
      std::cout << "Basic Execution time: " << duration.count()/1000000.0f << std::endl;
    #endif
    

    // Press  ESC on keyboard to exit
    char c=(char)waitKey(25);
    if(c==27)
      break;
  }
 
  // When everything done, release the video capture object
  cap.release();

  // Closes all the frames
  destroyAllWindows();
  
  return 0;
}

/* @function readme */
  void readme()
  { std::cout << " Usage: capture <deviceid>" << std::endl; }

void Kalman_Filter(Mat &x_k, Mat &P_k, const double u_k, const double z_k, const double sigma_P2, const double sigma_u2, const double sigma_M2, const double dt)
{
  if(x_k.type()!=CV_64FC1 || P_k.type()!=CV_64FC1 || x_k.size()!=Size(1,2) || P_k.size()!=Size(2,2) )
    throw invalid_argument("Inpur Error in Kalman filter, matrices size or type wrong!!");

  Mat F_k = Mat::zeros(2, 2, CV_64FC1);
  F_k.at<double>(0,0) = 1.0;
  F_k.at<double>(0,1) = dt;
  F_k.at<double>(1,0) = 0.0;
  F_k.at<double>(1,1) = 1.0;

  Mat Q_k = Mat::zeros(2, 2, CV_64FC1);
  Q_k.at<double>(0,0) = 1.0/3.0 * pow(dt, 3.0);
  Q_k.at<double>(0,1) = 1.0/2.0 * pow(dt, 2.0);
  Q_k.at<double>(1,0) = 1.0/2.0 * pow(dt, 2.0);
  Q_k.at<double>(1,1) = 1.0/1.0 * pow(dt, 1.0);

  Mat Qu_k = Mat::zeros(2, 2, CV_64FC1);
  Qu_k.at<double>(0,0) = 1.0/4.0 * pow(dt, 4.0);
  Qu_k.at<double>(0,1) = 1.0/2.0 * pow(dt, 3.0);
  Qu_k.at<double>(1,0) = 1.0/2.0 * pow(dt, 3.0);
  Qu_k.at<double>(1,1) = 1.0/1.0 * pow(dt, 2.0);

  
  // process update
  x_k = F_k * x_k;
  x_k.at<double>(0) +=  0.5*dt*dt * u_k;
  x_k.at<double>(1) +=  dt * u_k;

  P_k = F_k*P_k*F_k.t() + Q_k*sigma_P2 + Qu_k*sigma_u2; // sigma_P2 is continuous time white noise intensity

  // measurement matrix
  Mat H_k = Mat::zeros(1, 2, CV_64FC1);
  H_k.at<double>(0) = 1.0;


  // kalman filter gain (assumes H_k = [1, 0])
  Mat K_KF = Mat::zeros(2, 1, CV_64FC1); 
  double S_k = P_k.at<double>(0,0) + sigma_M2; // sigma_M2 is discrete time gaussian white noise variance
  K_KF.at<double>(0) = P_k.at<double>(0,0)/S_k;
  K_KF.at<double>(1) = P_k.at<double>(1,0)/S_k;

  // measurement update
  Mat temp = Mat::eye(2, 2, CV_64FC1) - K_KF*H_k;
  x_k = temp * x_k + K_KF * z_k;
  P_k = temp * P_k;
}

  

void Kalman_predict(Mat &x_k, Mat &P_k, const double u_k, const double sigma_P2, const double sigma_u2, const double dt)
{
  if(x_k.type()!=CV_64FC1 || P_k.type()!=CV_64FC1 || x_k.size()!=Size(1,2) || P_k.size()!=Size(2,2) )
    throw invalid_argument("Inpur Error in Kalman filter, matrices size or type wrong!!");

  Mat F_k = Mat::zeros(2, 2, CV_64FC1);
  F_k.at<double>(0,0) = 1.0;
  F_k.at<double>(0,1) = dt;
  F_k.at<double>(1,0) = 0.0;
  F_k.at<double>(1,1) = 1.0;

  Mat Q_k = Mat::zeros(2, 2, CV_64FC1);
  Q_k.at<double>(0,0) = 1.0/3.0 * pow(dt, 3.0);
  Q_k.at<double>(0,1) = 1.0/2.0 * pow(dt, 2.0);
  Q_k.at<double>(1,0) = 1.0/2.0 * pow(dt, 2.0);
  Q_k.at<double>(1,1) = 1.0/1.0 * pow(dt, 1.0);

  Mat Qu_k = Mat::zeros(2, 2, CV_64FC1);
  Qu_k.at<double>(0,0) = 1.0/4.0 * pow(dt, 4.0);
  Qu_k.at<double>(0,1) = 1.0/2.0 * pow(dt, 3.0);
  Qu_k.at<double>(1,0) = 1.0/2.0 * pow(dt, 3.0);
  Qu_k.at<double>(1,1) = 1.0/1.0 * pow(dt, 2.0);

  
  // process update
  x_k = F_k * x_k;
  x_k.at<double>(0) +=  0.5*dt*dt * u_k;
  x_k.at<double>(1) +=  dt * u_k;

  P_k = F_k*P_k*F_k.t() + Q_k*sigma_P2 + Qu_k*sigma_u2; // sigma_P2 is continuous time white noise intensity

}

