#include "opencv2/opencv.hpp"
#include <iostream>

using namespace std;
using namespace cv;

void readme();

int main( int argc, char** argv ){

  if (argc != 2 )
  { readme(); return -1; }

  // Create a VideoCapture object and open the input file
  // If the input is the web camera, pass 0 instead of the video file name

  printf("\n%d\n",stoi(argv[1]));
  VideoCapture cap(stoi(argv[1])); 
   
  // Check if camera opened successfully
  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }
  cap.set(CAP_PROP_FPS, 60.0); 
  cap.set(CAP_PROP_FRAME_WIDTH,1920);
  cap.set(CAP_PROP_FRAME_HEIGHT,1200);
  // cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G'));
  int img_width = cap.get(CAP_PROP_FRAME_WIDTH);
  int img_height = cap.get(CAP_PROP_FRAME_HEIGHT);

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

  //for displaying matches
  namedWindow("frame", WINDOW_NORMAL);
	
  while(1){

    Mat frame;
    // Capture frame-by-frame
    cap >> frame;
    if(frame.channels() == 3)
    {
      cvtColor(frame, frame, COLOR_BGR2GRAY);
    }

    
    Mat rectified_frame;
    undistort(frame, rectified_frame, K_matrix, distCoeffs);


    // Display the resulting frame
    imshow( "frame", rectified_frame);

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
