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

  if (argc != 3 )
  { readme(); return -1; }

  // image of model
  Mat img_object = imread( String("/home/aero/Desktop/cam_ws/src/Tracker/images/")+ String(argv[2]) + String(".png"), IMREAD_GRAYSCALE );
  if( !img_object.data)
  { cout<< " --(!) Error reading model image " << endl; return -1; }
  auto diag = sqrt(img_object.size[0]*img_object.size[0] + img_object.size[1]*img_object.size[1]);
  Mat img_scene;

  // Initialize surf detector and extract descriptors for the model image
  Ptr<ORB> detector = ORB::create( 10000 );
  std::vector<KeyPoint> keypoints_object, keypoints_scene;
  Mat descriptors_object, descriptors_scene;
  detector->detectAndCompute( img_object, Mat(), keypoints_object, descriptors_object );
  
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
   
  // Check if camera opened successfully
  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }
	
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
    
    std::vector<std::vector<cv::DMatch>> matches12, matches21;
    matcher.knnMatch( descriptors_object, descriptors_scene, matches12,2 );
    matcher.knnMatch( descriptors_scene, descriptors_object, matches21,2 );

    //-- Quick calculation of min distances between keypoints
    double min_dist12 = 50;
    for( int i = 0; i < matches12.size(); i++ )
    { 
      if( matches12[i].size()>=1)
      { 
        if( matches12[i][0].distance < min_dist12 ) min_dist12 = matches12[i][0].distance;
      }
    }

    double min_dist21 = 50;
    for( int i = 0; i < matches21.size(); i++ )
    { 
      if( matches21[i].size()>=1)
      { 
        if( matches21[i][0].distance < min_dist21 ) min_dist21 = matches21[i][0].distance;
      }
    }

    printf("-- Min dist : %f \n", min_dist12 );

    //-- Lowe's Ratio test and minimum distance threshold test 
    std::vector< DMatch > good_matches12, good_matches21;
    for( int i = 0; i < matches12.size(); i++ )
    { 
      if( matches12[i].size()>=2 && matches12[i][0].distance <= 10*min_dist12 && matches12[i][0].distance < matches12[i][1].distance * 0.8)
      { good_matches12.push_back( matches12[i][0]); }
    }

    for( int i = 0; i < matches21.size(); i++ )
    { 
      if( matches21[i].size()>=2 && matches21[i][0].distance <= 10*min_dist21 && matches21[i][0].distance < matches21[i][1].distance * 0.8)
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
    drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
                 best_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                 std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    
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

        std::vector< DMatch > filtered_best_matches;
        for ( int i =0; i<best_matches.size(); i++)
        {
          Point2f diff = scene_feature_loc[best_matches[i].queryIdx] - keypoints_scene[ best_matches[i].trainIdx ].pt;
          if(sqrt(diff.x*diff.x + diff.y*diff.y) < 10)
          {
            filtered_best_matches.push_back(best_matches[i]);
          }
        }

        if(filtered_best_matches.size() >= 1)// no. of matches to use for pose estimation
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
      // printf("No enough matches: %ld\n", best_matches.size());
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

