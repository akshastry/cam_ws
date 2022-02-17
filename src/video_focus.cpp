#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>

using namespace std;
using namespace cv;

void readme();
bool FftShift(const Mat&, Mat&);

int main( int argc, char** argv ){

  if (argc != 2 )
  { readme(); return -1; }

  // Create a VideoCapture object and open the input file
  // If the input is the web camera, pass 0 instead of the video file name

  printf("\n%d\n",stoi(argv[1]));
  VideoCapture cap(stoi(argv[1]));
  cap.set(cv::CAP_PROP_FPS, 60.0); 
  cap.set(cv::CAP_PROP_FRAME_WIDTH,1280);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT,720);
  cap.set(cv::CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G'));

  cout << "Brightness: " << cap.get(CAP_PROP_BRIGHTNESS) << "\n";
  cout << "Contrast: " << cap.get(CAP_PROP_CONTRAST) << "\n";
  cout << "Sharpness: " << cap.get(CAP_PROP_SHARPNESS) << "\n";
  cout << "Saturation: " << cap.get(CAP_PROP_SATURATION) << "\n";
  cout << "Gain: " << cap.get(CAP_PROP_GAIN) << "\n";
  cout << "Gamma: " << cap.get(CAP_PROP_GAMMA) << "\n";

  // Check if camera opened successfully
  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }
  cap.set(CAP_PROP_FRAME_WIDTH,1280);
  cap.set(CAP_PROP_FRAME_HEIGHT,720);
  cap.set(CAP_PROP_FPS,60);

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
 
    // If the frame is empty, break immediately
    if (frame.empty())
      break;

    Mat padded;                            //expand input image to optimal size
    int m = getOptimalDFTSize( frame.rows );
    int n = getOptimalDFTSize( frame.cols ); // on the border add zero values
    copyMakeBorder(frame, padded, 0, m - frame.rows, 0, n - frame.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

    dft(complexI, complexI, DFT_COMPLEX_OUTPUT);
    Mat frame_fftshift;
    FftShift(complexI, frame_fftshift);

    split(frame_fftshift, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    Mat magI = planes[0];

    double s;
    s = sum(magI)[0];
    magI = magI / s;
    Mat magI_log;
    log(magI, magI_log);
    double fde = sum(magI.mul( magI_log))[0];

    cout << "fde = " << fde << "\n";



    // Display the resulting frame
    imshow( "frame", frame );

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


  bool FftShift(const Mat& src, Mat& dst)
  {
    if(src.empty()) return true;

    const uint h=src.rows, w=src.cols;        // height and width of src-image
    const uint qh=h>>1, qw=w>>1;              // height and width of the quadrants

    Mat qTL(src, Rect(   0,    0, qw, qh));   // define the quadrants in respect to
    Mat qTR(src, Rect(w-qw,    0, qw, qh));   // the outer dimensions of the matrix.
    Mat qBL(src, Rect(   0, h-qh, qw, qh));   // thus, with odd sizes, the center
    Mat qBR(src, Rect(w-qw, h-qh, qw, qh));   // line(s) get(s) omitted.

    Mat tmp;
    hconcat(qBR, qBL, dst);                   // build destination matrix with switched
    hconcat(qTR, qTL, tmp);                   // quadrants 0 & 2 and 1 & 3 from source
    vconcat(dst, tmp, dst);

    return false;
  }