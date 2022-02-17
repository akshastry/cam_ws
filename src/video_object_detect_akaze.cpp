#include "opencv2/opencv.hpp"
#include <iostream>
#include <stdio.h>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

void readme();

int main( int argc, char** argv ){

  if (argc != 2 )
  { readme(); return -1; }

  // Create a VideoCapture object and open the input file
  // If the input is the web camera, pass 0 instead of the video file name
  VideoCapture cap(stoi(argv[1])); 
   
  // Check if camera opened successfully
  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }

  // image of model
  Mat img_object = imread( String("/home/aero/Desktop/cam_ws/src/Tracker/images/1.png"), IMREAD_GRAYSCALE );
  if( !img_object.data)
  { cout<< " --(!) Error reading model image " << endl; return -1; }
  Mat img_scene;

  // Initialize surf detector and extract descriptors for the model image
  // int minHessian = 400;
  Ptr<AKAZE> detector = AKAZE::create( );
  std::vector<KeyPoint> keypoints_object, keypoints_scene;
  Mat descriptors_object, descriptors_scene;
  detector->detectAndCompute( img_object, Mat(), keypoints_object, descriptors_object );
  // detector = SURF::create();
  

  //for displaying matches
  namedWindow("Good Matches & Object detection", WINDOW_NORMAL);
  
	// loop over input video images until esc pressed
  while(1){

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

    img_scene = frame;

    //-- Step 1: Detect the keypoints and extract descriptors
    detector->detectAndCompute( img_scene, Mat(), keypoints_scene, descriptors_scene );

    //-- Step 2: Matching descriptor vectors using FLANN or BruteForce matcher
    // FlannBasedMatcher matcher;
    BFMatcher matcher(NORM_HAMMING);
    
    std::vector<std::vector<cv::DMatch>> matches;
    matcher.knnMatch( descriptors_object, descriptors_scene, matches,2 );

    //-- Quick calculation of max and min distances between keypoints
    double max_dist = 0; double min_dist = 100;
    for( int i = 0; i < matches.size(); i++ )
    { 
      if( matches[i].size()>=1)
      { 
        double dist = matches[i][0].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
      }
    }
    // printf("-- Max dist : %f \n", max_dist );
    // printf("-- Min dist : %f \n", min_dist );

    //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
    std::vector< DMatch > good_matches;
    for( int i = 0; i < matches.size(); i++ )
    { 
      if( matches[i].size()>=2 && matches[i][0].distance <= 3*min_dist && matches[i][0].distance < matches[i][1].distance * 0.7)
      { good_matches.push_back( matches[i][0]); }
    }
    Mat img_matches = img_scene;
    drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
                 good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                 std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    
    if(good_matches.size() > 2)
    {
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

      if (! H.empty())
      {    
        perspectiveTransform( obj_corners, scene_corners, H);

        if(isContourConvex(scene_corners))
        {
          //-- Draw lines between the corners (the mapped object in the scene - image_2 )
          line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
          line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
          line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
          line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
        }

      }
    }
    else{
      printf("No enough matches: %ld\n", good_matches.size());
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

