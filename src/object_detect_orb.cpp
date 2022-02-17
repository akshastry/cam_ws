#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <chrono>
using namespace cv;
using namespace cv::xfeatures2d;
void readme();
/* @function main */
int main( int argc, char** argv )
{
  if( argc != 2 )
  { readme(); return -1; }

  // printf("%s \n", argv[1]);
  Mat img_object = imread( String("/home/aero/Desktop/cam_ws/src/Tracker/lor_images/1.jpg"), IMREAD_GRAYSCALE );
  Mat img_scene = imread( argv[1], IMREAD_GRAYSCALE );
  if( !img_object.data || !img_scene.data )
  { std::cout<< " --(!) Error reading images " << std::endl; return -1; }
  
  //-- Step 1: Detect the keypoints and extract descriptors using SURF
  int nfeatures = 500;
  Ptr<ORB> detector = ORB::create( nfeatures );
  std::vector<KeyPoint> keypoints_object, keypoints_scene;
  Mat descriptors_object, descriptors_scene;
  detector->detectAndCompute( img_object, Mat(), keypoints_object, descriptors_object );
  auto start = std::chrono::high_resolution_clock::now();
  detector->detectAndCompute( img_scene, Mat(), keypoints_scene, descriptors_scene );
  
  //-- Step 2: Matching descriptor vectors using FLANN matcher
  FlannBasedMatcher matcher = FlannBasedMatcher(makePtr<flann::LshIndexParams>(12, 20, 2));
  // std::vector< DMatch > matches;
  std::vector<std::vector<cv::DMatch>> matches;
  matcher.knnMatch( descriptors_object, descriptors_scene, matches,2 );
  
      // execution time
      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
      std::cout << "Basic Execution time: " << duration.count()/1000000.0f << std::endl;

  double max_dist = 0; double min_dist = 100;
  //-- Quick calculation of max and min distances between keypoints
  for( int i = 0; i < matches.size(); i++ )
  { 
    if(matches[i].size()>=1)
    { //std::cout << i << std::endl;
      double dist = matches[i][0].distance;
      // std::cout << i << std::endl;
      if( dist < min_dist ) min_dist = dist;
      if( dist > max_dist ) max_dist = dist;
    }
  }
  printf("-- Max dist : %f \n", max_dist );
  printf("-- Min dist : %f \n", min_dist );
  
  //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
  std::vector< DMatch > good_matches;
  for( int i = 0; i < matches.size(); i++ )
  { if( matches[i].size()>=2 && matches[i][0].distance <= 3*min_dist && matches[i][0].distance < matches[i][1].distance * 0.7 )
     { good_matches.push_back( matches[i][0]); }
  }
  Mat img_matches;
  drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
  
  //-- Localize the object
  std::vector<Point2f> obj;
  std::vector<Point2f> scene;
  for( size_t i = 0; i < good_matches.size(); i++ )
  {
    //-- Get the keypoints from the good matches
    obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
    scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
  }
  Mat H = findHomography( obj, scene, RANSAC );
  
  //-- Get the corners from the image_1 ( the object to be "detected" )
  std::vector<Point2f> obj_corners(4);
  obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
  obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
  std::vector<Point2f> scene_corners(4);
  perspectiveTransform( obj_corners, scene_corners, H);
  
  //-- Draw lines between the corners (the mapped object in the scene - image_2 )
  line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
  line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
  line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
  line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
  
  //-- Show detected matches
  namedWindow("Good Matches & Object detection", WINDOW_NORMAL);
  imshow( "Good Matches & Object detection", img_matches );
  waitKey(0);
  return 0;
  }
  
  /* @function readme */
  void readme()
  { std::cout << " Usage: ./SURF_descriptor <img1> <img2>" << std::endl; }