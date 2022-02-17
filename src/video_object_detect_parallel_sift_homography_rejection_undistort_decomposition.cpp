#include "opencv2/opencv.hpp"
#include <iostream>
#include <stdio.h>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"

#include <chrono>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

#define measure_exec_time

void readme();

int main( int argc, char** argv ){

  if (argc != 3 )
  { readme(); return -1; }

  // image of model
  Mat img_object = imread( String("/home/aero/Desktop/cam_ws/src/Tracker/images/")+ String(argv[2]) + String(".png"), IMREAD_GRAYSCALE );
  if( !img_object.data)
  { cout<< " --(!) Error reading model image " << endl; return -1; }
  Mat img_scene; // matrix to store scene image

  int img_width = img_object.size().width;
  int img_height = img_object.size().height;

  cout << "object_image WIDTH: " << img_width << "\n";
  cout << "object_image HEIGHT: " << img_height << "\n";


  // Initialize sift detector and extract descriptors for the model image
  Ptr<SIFT> detector = SIFT::create( );
  std::vector<KeyPoint> keypoints_object, keypoints_scene;
  Mat descriptors_object, descriptors_scene;
  detector->detectAndCompute( img_object, Mat(), keypoints_object, descriptors_object );
  cout << "No. of object image features: " << keypoints_object.size() << "\n";

  vector<Point2f> obj_feature_loc; // 
  vector<Point3d> obj_feature_loc_3D;
  float img_width_length = 25.8; // cm
  double img_height_length = 14.6; // cm
  for (int i = 0; i<keypoints_object.size(); i++)
  {
    obj_feature_loc.push_back(keypoints_object[i].pt);
    obj_feature_loc_3D.push_back(Point3f(keypoints_object[i].pt.y * img_height_length/img_height, keypoints_object[i].pt.x * img_width_length/img_width, 0.0));
  }
  

  //for displaying matches
  namedWindow("Good Matches & Object detection", WINDOW_NORMAL);
  

  // Create a VideoCapture object and open the input stream
  VideoCapture cap(stoi(argv[1])); 

  // Check if camera opened successfully
  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }

  // set camera properties
  cap.set(cv::CAP_PROP_FPS, 60.0); 
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);// calibration is with this value
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 360);// calibration is with this value
  cap.set(cv::CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G'));
  img_width = cap.get(CAP_PROP_FRAME_WIDTH);
  img_height = cap.get(CAP_PROP_FRAME_HEIGHT);

  cout << "scene_image WIDTH: " << img_width << "\n";
  cout << "scene_image HEIGHT: " << img_height << "\n";

  
  

  // intrinsic camera parameters
  //      [ fx   0  cx ]
  //      [  0  fy  cy ]
  //      [  0   0   1 ]
  Mat K_matrix = Mat::zeros(3, 3, CV_64FC1);
  K_matrix.at<double>(0, 0) = 964.8946/2.0; //fx, std_dev +- 0.6325     
  K_matrix.at<double>(1, 1) = 969.9548/2.0; //fy, std_dev +- 0.6160     
  K_matrix.at<double>(0, 2) = 614.5199/2.0; //cx, std_dev +- 0.8976     
  K_matrix.at<double>(1, 2) = 452.0068/2.0; //cy, std_dev +- 0.7236
  K_matrix.at<double>(2, 2) = 1;

  // radial distortion coefficients
  Mat distCoeffs = Mat::zeros(5, 1, CV_64FC1); 
  distCoeffs.at<double>(0) = -0.37491173; //k1, std_dev +- 0.00199622
  distCoeffs.at<double>(1) = +0.23393262; //k2, std_dev +- 0.00919023
  distCoeffs.at<double>(2) = -0.00222111; //p1, std_dev +- 0.00013275
  distCoeffs.at<double>(3) = +0.00183102; //p2, std_dev +- 0.00010322
  distCoeffs.at<double>(4) = -0.10912843; //k3, std_dev +- 0.01173290

  // RANSAC parameters
  int iterationsCount = 500;        // number of Ransac iterations.
  float reprojectionError = 2.0;    // maximum allowed distance to consider it an inlier.
  float confidence = 0.99;          // RANSAC successful confidence
  bool useExtrinsicGuess = false;

  // for solvePnP
  Mat rvec = Mat::zeros(3, 1, CV_64FC1);          // output rotation vector
  Mat tvec = Mat::zeros(3, 1, CV_64FC1);          // output translation vector
  Mat inliers_id;

  // 3D axes
  vector<Point3f> axes;
  axes.push_back(Point3f(0.0, 0.0, 0.0)); // origin
  axes.push_back(Point3f(1.0, 0.0, 0.0)); // X
  axes.push_back(Point3f(0.0, 1.0, 0.0)); // Y
  axes.push_back(Point3f(0.0, 0.0, 1.0)); // Z
  
	
  while(1){
    #if defined measure_exec_time
      auto start = std::chrono::high_resolution_clock::now();
    #endif

    // clear vectors
    keypoints_scene.clear();

    Mat frame;
    // Capture frame-by-frame
    cap >> frame;
 
    // If the frame is empty, break immediately
    if (frame.empty())
      break;

    // convert go grayscale
    if(frame.channels() == 3)
    {
      cvtColor(frame, frame, COLOR_BGR2GRAY);
    }



    //undistort frames
    Mat rectified_frame;
    undistort(frame, rectified_frame, K_matrix, distCoeffs);
    
    //resize to a smaller frame for speed
    int scale = 1; // scale down factor
    Mat final_frame;
    resize(rectified_frame, img_scene, Size(img_width/scale, img_height/scale), INTER_AREA);
    scale = 1;



    //-- Step 1: Detect the keypoints and extract descriptors
    detector->detectAndCompute( img_scene, Mat(), keypoints_scene, descriptors_scene );
    // cout << "No. of scene image features: " << keypoints_scene.size() << ", ";

    

    //-- Step 2: Matching descriptor vectors using FLANN or BruteForce matcher
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


    Mat img_matches = img_scene;
    

    int detected = false;
    vector< DMatch > filtered_best_matches;// container for final filtered matches    
    if(best_matches.size() >= 4)// no. of matches to use for pose estimation
    {
      //-- Localize the object
      vector<Point2f> obj;
      vector<Point2f> scene;
      for( size_t i = 0; i < best_matches.size(); i++ )
      {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints_object[ best_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_scene[ best_matches[i].trainIdx ].pt );
      }
      Mat H = findHomography( obj, scene, RANSAC );

      vector<Point2f> scene_feature_loc;
      if (! H.empty())
      { 

        perspectiveTransform(obj_feature_loc, scene_feature_loc, H);

        
        for ( int i =0; i<best_matches.size(); i++)
        {
          Point2f diff = scene_feature_loc[best_matches[i].queryIdx] - keypoints_scene[ best_matches[i].trainIdx ].pt;
          if(sqrt(diff.x*diff.x + diff.y*diff.y) < 2)
          {
            filtered_best_matches.push_back(best_matches[i]);
          }
        }

        if(filtered_best_matches.size() >= 4)// no. of matches to use for pose estimation
        {
          //-- Localize the object
          vector<Point3d> obj_3D;
          vector<Point2d> scene_2D;
          for( size_t i = 0; i < filtered_best_matches.size(); i++ )
          {
            //-- Get the keypoints from the good matches
            obj_3D.push_back( obj_feature_loc_3D[ filtered_best_matches[i].queryIdx ] );
            scene_2D.push_back( keypoints_scene[ filtered_best_matches[i].trainIdx ].pt );

            obj.push_back( keypoints_object[ filtered_best_matches[i].queryIdx ] );
            scene.push_back( keypoints_scene[ filtered_best_matches[i].trainIdx ].pt);
          }

          solvePnPRansac( obj_3D, scene_2D, K_matrix, Mat::zeros(5, 1, CV_64FC1), rvec, tvec,
                  useExtrinsicGuess, iterationsCount, reprojectionError, confidence,
                  inliers_id, SOLVEPNP_IPPE );

          useExtrinsicGuess = true;

          // project axes on image to draw
          vector<Point2f> axes2D;
          projectPoints(axes, rvec, tvec, K_matrix, Mat::zeros(5, 1, CV_64FC1), axes2D);
          for (int i = 0; i < 4; i++)
            axes2D[i] = axes2D[i] / (1.0* scale);

          // // get the corners
          vector<Point3f> obj_corners(4);
          obj_corners[0] = Point3f(0, 0, 0); obj_corners[1] = Point3f( img_height_length, 0, 0);
          obj_corners[2] = Point3f(img_height_length, img_width_length, 0); obj_corners[3] = Point3f( 0, img_width_length, 0);
          vector<Point2f> scene_corners(4);
          projectPoints(obj_corners, rvec, tvec, K_matrix, Mat::zeros(5, 1, CV_64FC1), scene_corners);
          for (int i = 0; i < 4; i++)
            scene_corners[i] = scene_corners[i] / (1.0* scale);

          // draw matches
          drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
           filtered_best_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
           std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

          //-- Draw lines between the corners (the mapped object in the scene - image_2 )
          line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(255, 0, 0), 4 );
          line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar(255, 0, 0), 4 );
          line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar(255, 0, 0), 4 );
          line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar(255, 0, 0), 4 );

          line( img_matches, axes2D[0] + Point2f( img_object.cols, 0), axes2D[1] + Point2f( img_object.cols, 0), Scalar(255, 0, 0), 4 );
          line( img_matches, axes2D[0] + Point2f( img_object.cols, 0), axes2D[2] + Point2f( img_object.cols, 0), Scalar( 0, 0, 255), 4 );
          line( img_matches, axes2D[0] + Point2f( img_object.cols, 0), axes2D[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );

          detected = true;

          H = findHomography( obj, scene, RANSAC );


          if(!H.empty())
          {

            //-- Get the corners from the image_1 ( the object to be "detected" )
            vector<Point2f> obj_corners2D(4);
            obj_corners2D[0] = Point(0,0); obj_corners2D[1] = Point( img_object.cols, 0 );
            obj_corners2D[2] = Point( img_object.cols, img_object.rows ); obj_corners2D[3] = Point( 0, img_object.rows );
            vector<Point2f> scene_corners2D(4);


            perspectiveTransform( obj_corners2D, scene_corners2D, H);

            if(isContourConvex(scene_corners))
            {
              // detected = true;
              
              // // draw matches
              // drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
              //    filtered_best_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
              //    std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

              //-- Draw lines between the corners (the mapped object in the scene - image_2 )
              line( img_matches, scene_corners2D[0] + Point2f( img_object.cols, 0), scene_corners2D[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
              line( img_matches, scene_corners2D[1] + Point2f( img_object.cols, 0), scene_corners2D[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
              line( img_matches, scene_corners2D[2] + Point2f( img_object.cols, 0), scene_corners2D[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
              line( img_matches, scene_corners2D[3] + Point2f( img_object.cols, 0), scene_corners2D[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
            }
          }
        }
      }
    }
    else{
      // printf("No enough matches: %ld\n", good_matches.size());
    }

    if(detected == false){
      drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
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

