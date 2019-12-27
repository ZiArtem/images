#include "Filters.h"

using namespace std;
using namespace cv;

struct regions {
  int n;
  int m;
  int count;
  int** arr;
  std::vector<float> regionSize;
  std::vector<float> regionSumIntensity;

  regions(int _n, int _m) {
    count = 0;
    n = _n;
    m = _m;
    arr = new int* [n];
    for (int i = 0; i < n; ++i) arr[i] = new int[m];

    for (int i = 0; i < n; ++i)
      for (int j = 0; j < m; ++j) arr[i][j] = 0;

    regionSize.push_back(0);
    regionSumIntensity.push_back(0);
  }

  ~regions() {
    for (int i = 0; i < n; ++i) delete[] arr[i];
    delete[] arr;
  }

  void createRegion(int intensity, int x, int y) {
    count++;
    arr[x][y] = count;
    regionSize.push_back(1.);
    regionSumIntensity.push_back(intensity);
  }

  void addElem(int intensity, int x, int y, int region) {
    arr[x][y] = region;
    regionSize[region] += 1;
    regionSumIntensity[region] += intensity;
  }

  double clavg(int region) {
    if (region) return (regionSumIntensity[region] / regionSize[region]);
    return 0;
  }

  void merge(int b, int c, int intensity, int x, int y) {
    // c < b
    if (c != b) {
      regionSumIntensity[c] += regionSumIntensity[b];
      regionSize[c] += (regionSize[b] + 1);

      for (int i = 0; i <= x; ++i)
        for (int j = 0; j <= y; ++j)
          if (arr[i][j] == b) arr[i][j] = c;

      count--;
    }
    this->addElem(intensity, x, y, c);
  }
};

void gray(const Mat& input, Mat& output) {
  for (int i = 0; i < input.rows; ++i)
    for (int j = 0; j < input.cols; ++j) {
      Vec3b colorIn = input.at< Vec3b>(i, j);
      unsigned short colorOut = colorIn[0] * 0.0721 + colorIn[1] * 0.7154 + colorIn[2] * 0.2125;
      output.at< Vec3b>(i, j) = Vec3b(colorOut, colorOut, colorOut);
    }
}

unsigned char OtsuThreshold(const Mat& input) {
  int hist[256];
  int sumOfIntensity = 0;
  for (int i = 0; i < 256; ++i) hist[i] = 0;
  for (int i = 0; i < input.rows; ++i)
    for (int j = 0; j < input.cols; ++j) {
      sumOfIntensity += input.at< Vec3b>(i, j)[0];
      hist[input.at< Vec3b>(i, j)[0]]++;
    }

  int pixelCount = input.rows * input.cols;
  int bestThresh = 0;
  double bestSigma = 0.0;
  int firstClassPixelCount = 0;
  int64_t firstClassIntensitySum = 0;
  for (int i = 0; i < 255; ++i) {
    firstClassPixelCount += hist[i];
    firstClassIntensitySum += i * hist[i];
    double firstClassProb = firstClassPixelCount / static_cast<double>(pixelCount);
    double secondClassProb = 1.0 - firstClassProb;
    double firstClassMean = firstClassIntensitySum / static_cast<double>(firstClassPixelCount);
    double secondClassMean = (sumOfIntensity - firstClassIntensitySum) /
      static_cast<double>(pixelCount - firstClassPixelCount);
    double meanDelta = firstClassMean - secondClassMean;
    double sigma = firstClassProb * secondClassProb * pow(meanDelta, 2);
    if (sigma > bestSigma) {
      bestSigma = sigma;
      bestThresh = i;
    }
  }
  return bestThresh;
}

void Binarization(const Mat& input, Mat& output, unsigned char threshold) {
  for (int i = 0; i < input.rows; ++i)
    for (int j = 0; j < input.cols; ++j) {
      unsigned char currColor = input.at< Vec3b>(i, j)[0];
      unsigned char newColor;
      newColor = currColor < threshold ? 0 : 255;
      output.at< Vec3b>(i, j) = Vec3b(newColor, newColor, newColor);
    }
}

void OtsuFilter(const Mat& input, Mat& output) {
  gray(input, output);

  unsigned char threshold = OtsuThreshold(output);
  int tresh = static_cast<int>(threshold);
  Mat newInput;
  output.copyTo(newInput);
  Binarization(newInput, output, threshold);
  RegionGrowing(output, 0.1);
}

void RegionGrowing(Mat& source, double sigma) {
  regions r(source.rows, source.cols);
  double deltaB, deltaC, B, C;
  for (int x = 1; x < source.rows; ++x)
    for (int y = 1; y < source.cols; ++y) {
      int intensity = source.at<uchar>(x, y) ? 1 : 0;
      if (intensity > 0) {
        if (x == 1 && y == 1)
          B = C = 0;
        else {
          B = r.clavg(r.arr[x][y - 1]);
          C = r.clavg(r.arr[x - 1][y]);
        }
        intensity = 1;
        deltaB = abs(intensity - B);
        deltaC = abs(intensity - C);
        if (deltaB > sigma&& deltaC > sigma) r.createRegion(intensity, x, y);
        if (deltaB <= sigma && deltaC > sigma)
          r.addElem(intensity, x, y, r.arr[x][y - 1]);
        if (deltaB > sigma&& deltaC <= sigma)
          r.addElem(intensity, x, y, r.arr[x - 1][y]);
        if (deltaB <= sigma && deltaC <= sigma)
          if (abs(B - C) <= sigma)
            r.merge(r.arr[x][y - 1], r.arr[x - 1][y], intensity, x, y);
          else if (deltaB < deltaC)
            r.addElem(intensity, x, y, r.arr[x][y - 1]);
          else
            r.addElem(intensity, x, y, r.arr[x - 1][y]);
      }
    }
  cout << "Count regions = " << r.count << endl;
}