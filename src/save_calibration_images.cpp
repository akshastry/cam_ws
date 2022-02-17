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
	int img_cnt = 0;
  while(1){

    Mat frame;
    // Capture frame-by-frame
    cap >> frame;
    if(frame.channels() == 3)
    {
      cvtColor(frame, frame, CV_BGR2GRAY);
    }
    // cvtColor(frame, frame, CV_BGR2GRAY);
 
    // If the frame is empty, break immediately
    if (frame.empty())
      break;

    // Display the resulting frame
    imshow( "Frame", frame );

    // Press  ESC on keyboard to exit
    char c=(char)waitKey(25);
    if(c==27)
      break;
    else if(c==13)
    {
      printf("%s\n",("Saving image...... /home/aero/Desktop/cam_ws/src/Tracker/calibration_images/"+to_string(img_cnt)+".png").c_str());
      imwrite("/home/aero/Desktop/cam_ws/src/Tracker/calibration_images/"+to_string(img_cnt)+".png", frame);
      img_cnt++;
    }
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
