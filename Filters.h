#pragma once
#include <ctime>
#include <random>
#include <math.h>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

void gray(const Mat& input, Mat& output);
unsigned char OtsuThreshold(const Mat& input);
void Binarization(const Mat& input, Mat& output, unsigned char threshold);
void OtsuFilter(const Mat& input, Mat& output);
void RegionGrowing(Mat& source, double sigma);
