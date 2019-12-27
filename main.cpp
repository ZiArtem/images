#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "Filters.h"

using namespace std;
using namespace cv;

int main() {
  Mat  original_image, res_image, fil_image;
  original_image = imread("G:/фот/Новая папка/aaa.jpg", 1);

  resize(original_image, res_image, res_image.size(), 0.7, 0.7);
  res_image.copyTo(fil_image);

  imshow("original image", res_image);
  OtsuFilter(res_image, fil_image);

  imshow("otsy image", fil_image);

  waitKey(0);
  return 0;
}