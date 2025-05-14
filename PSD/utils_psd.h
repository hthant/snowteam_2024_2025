#ifndef __UTILS_H_INCLUDED__
#define __UTILS_H_INCLUDED__

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <tuple>

using namespace std;
using namespace cv;
using ParticleSizeResult = tuple<
    vector<int>,    // area_pixels
    vector<double>, // diam_eq_mm
    vector<string>, // minutes
    vector<int>     // flakes
>;

pair<double, double> calculatePreConstants(double fovx, double fovy, int px_x, int px_y);

double findFov(int px, double f);

Mat computeProjection(const Mat& A, const Mat& R, const Mat& t);

pair<Mat, Mat> computeReverseRT(const Mat& R, const Mat& t);

int findNth(const string& haystack, const string& needle, int n);

Vec4d triangulatePoint(const Vec3d& x1, const Vec3d& x2, const Mat& P1, const Mat& P2);

ParticleSizeResult findParticleSizes(const vector<pair<string, string>>& selHour,
    const Mat& P1,
    const Mat& P0,
    double pre_x,
    double pre_y,
    const Mat& F10,
    const Vec3d& t0);

double calcSingleCamVolume();

void snowflakeBins();

#endif