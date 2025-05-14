#ifndef __MAIN_PSD__
#define __MAIN_PSD__

#include "utils_psd.cc"

using namespace std;
using namespace cv;

void big_main(int focus, int px_x, int px_y, double binsize = 0.25, int maxbin = 25) {
    // utp.extractCalibrationMatrices2 done in python.
    Mat A1 = (Mat_<double>(3,3) <<
        2805.20873, 0.0, 604.861162,
        0.0, 2799.34413, 956.665011,
        0.0, 0.0, 1.0);
    Mat A0 = (Mat_<double>(3,3) <<
        4885.09256, 0.0, 1020.34890,
        0.0, 4908.38966, 1298.92303,
        0.0, 0.0, 1.0);
    Mat R0 = (Mat_<double>(3,3) <<
        0.99752094,  0.05019746, -0.04931732,
        0.0163409, 0.51643399 , 0.85617108,
        0.06844675, -0.85485446,  0.51433344);
    Mat t0 = (Mat_<double>(1,3) <<
        21.82130914, -132.07328541,84.39935036);

    pair<Mat, Mat> R01_t01 = computeReverseRT(R0, t0);
    Mat R01 = R01_t01.first;
    Mat t01 = R01_t01.second;
    Mat P1 = computeProjection(A1, R01, t01);

    Mat R = Mat::eye(3, 3, CV_64F);
    Mat t = Mat::zeros(3, 1, CV_64F);
    Mat P0 = computeProjection(A0, R, t);

    // Fundamental Matrix from Matching
    Mat F;

    cout << t0 << endl;

    double ang_x = findFov(px_x, A0.at<double>(0,0));
    double ang_y = findFov(px_y, A0.at<double>(1,1));

    // CalculateSingleCamVolume
        // Will use -> int focus
    // Print Volume

    // From Matching
    vector<pair<string, string>> hr28;

    pair<double,double> pre_xy = calculatePreConstants(ang_x, ang_y, px_x, px_y);
    double pre_x = pre_xy.first;
    double pre_y = pre_xy.second;
    ParticleSizeResult test = findParticleSizes(hr28, P1, P0, pre_x, pre_y, F, t0);
}

int main() {
    int focus = 210;
    int px_x = 2448;
    int px_y = 2048;

    big_main(focus, px_x, px_y);
}

#endif