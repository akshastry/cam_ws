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
#include <armadillo>

#include <chrono>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    arma::mat A(4,4, arma::fill::zeros);
    A(0,1) = 1.0;
    A(1,3) = 1.0;
    A(2,3) = 1.0;
    A(3,2) = -1.0;
    A(3,3) = -0.8944;

    arma::mat Fk_arma = arma::expmat(A*2.0);
    Fk_arma = Fk_arma.t();

    Mat F_k1(4, 4, CV_64FC1, Fk_arma.memptr());

    cout << F_k1 << "\n";

    arma::mat Lambda(8,8, arma::fill::zeros);
    Lambda(arma::span(0,3), arma::span(0,3)) = -A;
    Lambda(arma::span(4,7), arma::span(4,7)) = A.t();
    Lambda(3,7) = 1.0;

    arma::mat eLT = arma::expmat(Lambda*3.0);
    arma::mat Qk_arma = Fk_arma.t() * eLT(arma::span(0,3), arma::span(4,7));

    Qk_arma = Qk_arma.t();
    Mat Q_k1(4, 4, CV_64FC1, Qk_arma.memptr());

    cout << Q_k1 << "\n";

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Basic Execution time: " << duration.count()/1000000.0f << std::endl;

}