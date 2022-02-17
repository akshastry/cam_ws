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


#include "GaussianBlur.h"
#include "Keypoint.h"
#include "LoG.h"
#include "Image.h"
#include "general_helpers.h"
#include <omp.h>

#include <vector>
#include <iostream>
#include <string>
#include <getopt.h>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

#define measure_exec_time

void readme();
double find_keypoint_features(Image & src, cv::Mat & result_features, 
    std::vector<cv::KeyPoint> & cv_keypoints);


bool debug = false;
int view_index = 0;
float grad_threshold = 0.0;
float intensity_threshold = 1;



int main( int argc, char** argv ){

  if (argc != 3 )
  { readme(); return -1; }

  // image of model
  Mat img_object = imread( String("/home/aero/Desktop/cam_ws/src/Tracker/images/")+ String(argv[2]) + String(".png"), IMREAD_GRAYSCALE );
  if( !img_object.data)
  { cout<< " --(!) Error reading model image " << endl; return -1; }
  // auto diag = sqrt(img_object.size[0]*img_object.size[0] + img_object.size[1]*img_object.size[1]);
  Mat img_scene;

  // Initialize surf detector and extract descriptors for the model image
  Ptr<SIFT> detector = SIFT::create( );
  std::vector<KeyPoint> keypoints_object, keypoints_scene;
  Mat descriptors_object, descriptors_scene;
  detector->detectAndCompute( img_object, Mat(), keypoints_object, descriptors_object );
  cout << "No. of object image features: " << keypoints_object.size() << "\n";

  vector<Point2f> obj_feature_loc; // 
  for (int i = 0; i<keypoints_object.size(); i++)
  {
    obj_feature_loc.push_back(keypoints_object[i].pt);
  }


  //for displaying matches
  namedWindow("Good Matches & Object detection", WINDOW_NORMAL);
  

  // Create a VideoCapture object and open the input file
  // If the input is the web camera, pass 0 instead of the video file name
  VideoCapture cap(stoi(argv[1])); 

  cap.set(CAP_PROP_FPS, 60.0); 
  cap.set(CAP_PROP_FRAME_WIDTH,1280);
  cap.set(CAP_PROP_FRAME_HEIGHT,720);
  cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G'));
  int img_width = cap.get(CAP_PROP_FRAME_WIDTH);
  int img_height = cap.get(CAP_PROP_FRAME_HEIGHT);

  cout << "WIDTH: " << img_width << "\n";
  cout << "HEIGHT: " << img_height << "\n";

  // intrinsic camera parameters
  //      [ fx   0  cx ]
  //      [  0  fy  cy ]
  //      [  0   0   1 ]
  Mat K_matrix = Mat::zeros(3, 3, CV_64FC1);
  K_matrix.at<double>(0, 0) = 964.8946; //fx, std_dev +- 0.6325     
  K_matrix.at<double>(1, 1) = 969.9548; //fy, std_dev +- 0.6160     
  K_matrix.at<double>(0, 2) = 614.5199; //cx, std_dev +- 0.8976     
  K_matrix.at<double>(1, 2) = 452.0068; //cy, std_dev +- 0.7236
  K_matrix.at<double>(2, 2) = 1;

  // radial distortion coefficients
  Mat distCoeffs = Mat::zeros(5, 1, CV_64FC1); 
  distCoeffs.at<double>(0) = -0.37491173; //k1, std_dev +- 0.00199622
  distCoeffs.at<double>(1) = +0.23393262; //k2, std_dev +- 0.00919023
  distCoeffs.at<double>(2) = -0.00222111; //p1, std_dev +- 0.00013275
  distCoeffs.at<double>(3) = +0.00183102; //p2, std_dev +- 0.00010322
  distCoeffs.at<double>(4) = -0.10912843; //k3, std_dev +- 0.01173290
   
  // Check if camera opened successfully
  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }
  
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
      cvtColor(frame, frame, CV_BGR2GRAY);
    }



    //undistort frames
    Mat rectified_frame;
    undistort(frame, rectified_frame, K_matrix, distCoeffs);
    
    //resize to a smaller frame for speed
    Mat final_frame;
    resize(rectified_frame, img_scene, Size(640, 360), INTER_AREA);



    //-- Step 1: Detect the keypoints and extract descriptors
    // detector->destectAndCompute( img_scene, Mat(), keypoints_scene, descriptors_scene );
    Image src2(img_scene);
    find_keypoint_features(src2, descriptors_scene, keypoints_scene);

    if(keypoints_scene.empty())
      continue;

    cout << "No. of scene image features: " << keypoints_scene.size() << ", ";

    #if defined measure_exec_time
    // execution time
      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
      std::cout << "Basic Execution time: " << duration.count()/1000000.0f << std::endl;
    #endif

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
    if(best_matches.size() >= 1)// no. of matches to use for pose estimation
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
          if(sqrt(diff.x*diff.x + diff.y*diff.y) < 10)
          {
            filtered_best_matches.push_back(best_matches[i]);
          }
        }

        if(filtered_best_matches.size() >= 4)// no. of matches to use for pose estimation
        {
          //-- Localize the object
          obj.clear();
          scene.clear();
          for( size_t i = 0; i < filtered_best_matches.size(); i++ )
          {
            //-- Get the keypoints from the good matches
            obj.push_back( keypoints_object[ filtered_best_matches[i].queryIdx ].pt );
            scene.push_back( keypoints_scene[ filtered_best_matches[i].trainIdx ].pt );
          }
          H = findHomography( obj, scene, RANSAC );

          if(!H.empty())
          {

            //-- Get the corners from the image_1 ( the object to be "detected" )
            vector<Point2f> obj_corners(4);
            obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
            obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
            vector<Point2f> scene_corners(4);


            perspectiveTransform( obj_corners, scene_corners, H);

            if(isContourConvex(scene_corners))
            {
              detected = true;
              
              // draw matches
              drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
                 best_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                 std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

              //-- Draw lines between the corners (the mapped object in the scene - image_2 )
              line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
              line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
              line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
              line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
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


double find_keypoint_features(Image & src, cv::Mat & result_features, 
        std::vector<cv::KeyPoint> & cv_keypoints) {
    ///////////////////////////////////// Algorithm BEGIN /////////////////////////////////////

    ///////////////////////////////////// LoG BEGIN /////////////////////////////////////
    // Find Difference of Gaussian Images using LoG
    LoG LoG_processor(src);
    std::vector<Image> octave1_log, octave2_log, octave3_log, octave4_log;

    LoG_processor.find_LoG_images(
        octave1_log, octave2_log, octave3_log, octave4_log);


    ///////////////////////////////////// Keypoint begin /////////////////////////////////////
    // Find keypoint image-pairs between the DoG images
    Keypoint kp_finder(src, grad_threshold, intensity_threshold);
    std::vector<Image> octave1_kp, octave2_kp, octave3_kp, octave4_kp;

    kp_finder.find_keypoints(octave1_log, octave1_kp);
    
    kp_finder.find_keypoints(octave2_log, octave2_kp);
    
    kp_finder.find_keypoints(octave3_log, octave3_kp);
    
    kp_finder.find_keypoints(octave4_log, octave4_kp);
    

    if (debug) cout << "Storing result" << endl;
    printf("%lu, %d\n", octave1_kp.size(), view_index);

    Image gradx(src.rows, src.cols), grady(src.rows, src.cols);
    std::vector<coord> keypoints;

    std::vector<PointWithAngle> points_with_angle;
    kp_finder.find_corners_gradients(octave1_kp[view_index], keypoints, points_with_angle);

    std::vector<float> kp_gradients;

    std::vector<KeypointFeature> keypoint_features;
    kp_finder.find_keypoint_orientations(keypoints, points_with_angle, 
        keypoint_features, src.rows, src.cols, standard_variances[2]);

    kp_finder.store_keypoints(keypoint_features, cv_keypoints, 1, src.cols);

    kp_finder.store_features(keypoint_features, result_features);
    
    return 0.0;
}

