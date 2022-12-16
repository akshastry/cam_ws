#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"

#include <iostream>
#include <string.h>
#include <cmath>
#include <fstream>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

void readme();
Mat crossmat(double x, double y, double z);
Mat eul2rotmat(double x, double y, double z);

int main( int argc, char** argv ){

  if (argc != 1 )
  { readme(); return -1; }

  cout << "OpenCV version: " << CV_VERSION << endl;

  //get the list of images from directory
  vector<String> fn;
  glob("/home/aero/Downloads/SFM_images_0/*.png", fn, false);

  // sort the images according to numeric file name
  sort(fn.begin(), fn.end(), [] (const string &first, const string &second){
        auto i = first.rfind("/");
        auto j = first.rfind(".png");
        int fi = stoi(first.substr(i+1, j-i-1));

        i = second.rfind("/");
        j = second.rfind(".png");
        int se = stoi(second.substr(i+1, j-i-1));
        return fi < se;
    
      });

  // // display the filenames
  // for(auto x:fn) cout << x << endl;

  //store the numbered image list
  vector<int> image_list;

  for(auto x:fn){
    auto i = x.rfind("/");
    auto j = x.rfind(".png");
    int fi = stoi(x.substr(i+1, j-i-1));

    image_list.push_back(fi);
  }

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


  // read first two images
  Mat raw_image1 = imread(fn[0]);
  Mat raw_image2 = imread(fn[1]);

  if(raw_image1.channels() == 3)
  {
    cvtColor(raw_image1, raw_image1, COLOR_BGR2GRAY);
  }

  if(raw_image2.channels() == 3)
  {
    cvtColor(raw_image2, raw_image2, COLOR_BGR2GRAY);
  }

  // read first two images
  Mat image1, image2;

  // undistort the images
  undistort(raw_image1, image1, K_matrix, distCoeffs);
  undistort(raw_image2, image2, K_matrix, distCoeffs);

  // cv::imshow("image1", image1);
  // cv::imshow("image2", image2);
  // // created the window by name image1
  // cv::waitKey(0);
  // cv::destroyWindow("image1");
  // cv::destroyWindow("image2");

  // create SIFT detector
  Ptr<SIFT> detector = SIFT::create( );
  
  // get SIFT keypoints and descriptors of the two images
  vector<KeyPoint> keypoints1, keypoints2;
  Mat descriptors1, descriptors2;
  detector->detectAndCompute( image1, Mat(), keypoints1, descriptors1 );
  detector->detectAndCompute( image2, Mat(), keypoints2, descriptors2 );

  // create best-first matcher based on L2 norm
  BFMatcher matcher(NORM_L2, true);
    
  // match the keypoints
  vector<DMatch> matches;
  matcher.match( descriptors1, descriptors2, matches);

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
  // E_mat = E_mat / E_mat.at<double>(1,2);

  // read the csv file and find the position and orientation of camera at the two images
  ifstream CSVfile ("/home/aero/Downloads/SFM_images_0/SFM_data.csv", ifstream::in);
  string line;

  getline(CSVfile, line);

  int image_no = 0;
  double T_x, T_y, T_z, roll, pitch, yaw, T_x1, T_y1, T_z1, roll1, pitch1, yaw1, T_x2, T_y2, T_z2, roll2, pitch2, yaw2;
  while(image_no != image_list[0])
  {
    getline(CSVfile, line);
    sscanf(line.c_str(), "%d,%lf,%lf,%lf,%lf,%lf,%lf\n", &image_no, &T_x1, &T_y1, &T_z1, &roll1, &pitch1, &yaw1);
  }
  cout << "image1 no.:" << image_no << endl;
  while(image_no != image_list[1])
  {
    getline(CSVfile, line);
    sscanf(line.c_str(), "%d,%lf,%lf,%lf,%lf,%lf,%lf\n", &image_no, &T_x2, &T_y2, &T_z2, &roll2, &pitch2, &yaw2);
  }
  CSVfile.close();
  cout << "image2 no.:" << image_no << endl;

  T_x = T_x2 - T_x1;
  T_y = T_y2 - T_y1;
  T_z = T_z2 - T_z1;
  Mat T = Mat::zeros(3, 1, CV_64FC1);
  T.at<double>(0,0) = T_x;
  T.at<double>(1,0) = T_y;
  T.at<double>(2,0) = T_z;

  Mat R1 = eul2rotmat(roll1, pitch1, yaw1);
  Mat R2 = eul2rotmat(roll2, pitch2, yaw2);
  double pi = 4.0*atan(1.0);
  Mat R_yaw180 = eul2rotmat(0.0, 0.0, pi);

  Mat R21 = R_yaw180 * R2 * R1.t() * R_yaw180.t();

  T = R1.t() * R_yaw180.t() * T;
  Mat cross_T  = crossmat(T.at<double>(0,0), T.at<double>(0,0), T.at<double>(0,0));

  Mat E_mat2 = cross_T * R21;
  E_mat2 = E_mat2 / E_mat2.at<double>(1,2);

  // cout << E_mat << endl;
  // cout << E_mat2 << endl;

  cout << Rot1 << endl;
  cout << Rot2 << endl;
  cout << t << endl;

  cout << R21 << endl;
  cout << T/norm(T) << endl;

  cout << eul2rotmat(pi/2, 0.0, pi/2) << endl;



  // vector<Mat> images;
  // size_t count = fn.size(); //number of png files in images folder
  // for (size_t i=0; i<count; i++)
  //     images.push_back(imread(fn[i]));
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

  Mat eul2rotmat(double r, double p, double y)
  {
    Mat ret = Mat(3, 3, CV_64FC1);

    ret.at<double>(0,0) = cos(y)*cos(p);
    ret.at<double>(1,0) = sin(y)*cos(p);
    ret.at<double>(2,0) = -sin(p);

    ret.at<double>(0,1) = cos(y)*sin(p)*sin(r) - sin(y)*cos(r);
    ret.at<double>(1,1) = sin(y)*sin(p)*sin(r) + cos(y)*cos(r);
    ret.at<double>(2,1) = cos(p)*sin(r);

    ret.at<double>(0,2) = cos(y)*sin(p)*cos(r) + sin(y)*sin(r);
    ret.at<double>(1,2) = sin(y)*sin(p)*cos(r) - cos(y)*sin(r);
    ret.at<double>(2,2) = cos(p)*cos(r);

    return ret;
  }

  /* @function readme */
  void readme()
  { std::cout << " Usage: capture <deviceid>" << std::endl; }