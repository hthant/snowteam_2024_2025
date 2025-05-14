#ifndef __UTILS_PSD_CC_INCLUDED__
#define __UTILS_PSD_CC_INCLUDED__

#include "utils_psd.h"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <utility>
#include <vector>
#include <string>
#include <tuple>
#include <filesystem>

using namespace std;
using namespace cv;

constexpr double pi = 3.141592653589793;

// Constant calculation using degrees to save some time
pair<double, double> calculatePreConstants(double fovx, double fovy, int px_x, int px_y) {
    // Convert degrees to radians
    fovx = fovx * pi / 180.0;
    fovy = fovy * pi / 180.0;
    
    double pre_x = tan(fovx / 2.0) * 2.0 / px_x;
    double pre_y = tan(fovy / 2.0) * 2.0 / px_y;
    
    return make_pair(pre_x, pre_y);
}

/*
    inputs:
        px - Pixels count in x or y direction
        f - Focal length in milliPixels
    outputs:
        returns the FOV angle in that direction
*/
double findFov(int px, double f) {
    return 2*atan(px / 2.0 / f) * 180.0 / pi;
}

// Compute projection matrix from camera intrinsics, rotation and translation
Mat computeProjection(const Mat& A, const Mat& R, const Mat& t) {
    // Reshape rotation matrix to 3x3 if needed
    Mat R_reshaped = R.reshape(1, 3); // 1 channel, 3 rows, 3 cols
    
    // Reshape translation vector to 3x1 if needed
    Mat t_reshaped = t.reshape(1, 3); // 1 channel, 3 rows, 1 col
    
    // Create the [R|t] matrix by horizontally concatenating R and t
    Mat Rt;
    hconcat(R_reshaped, t_reshaped, Rt);
    
    // Compute A * [R|t]
    return A * Rt;
}

// Compute reverse rotation and translation
pair<Mat, Mat> computeReverseRT(const Mat& R, const Mat& t) {
    // Reshape matrices
    Mat R_reshaped = R.reshape(1, 3);
    Mat t_reshaped = t.reshape(1, 3);
    
    // Create [R|t] matrix
    Mat Rt;
    hconcat(R_reshaped, t_reshaped, Rt);
    
    // Create padding row [0,0,0,1]
    Mat pad = (Mat_<double>(1, 4) << 0, 0, 0, 1);
    
    // Stack Rt and padding
    Mat Rtp;
    vconcat(Rt, pad, Rtp);
    
    // Compute inverse
    Mat reverse = Rtp.inv();
    
    // Extract rotation and translation  //Rect(x, y, width, height)
    Mat RR = reverse(Rect(0, 0, 3, 3));  // First 3x3 block
    Mat RT = reverse(Rect(3, 0, 1, 3));  // Last column, first 3 rows
    
    return make_pair(RR, RT);
}

// Helper function to find nth occurrence of a substring
int findNth(const string& haystack, const string& needle, int n) {
    size_t pos = 0;
    int count = 0;

    while (count < n && pos != string::npos) {
        pos = haystack.find(needle, pos + (count > 0 ? needle.length() : 0));
        if (pos != string::npos) count++;
    }
    return pos != string::npos ? static_cast<int>(pos) : -1;
}

// Point triangulation helper function
Vec4d triangulatePoint(const Vec3d& x1, const Vec3d& x2, const Mat& P1, const Mat& P2) {
    Mat M = Mat::zeros(6, 6, CV_64F);
    
    // Fill projection matrices
    P1.copyTo(M(Rect(0, 0, 4, 3)));
    P2.copyTo(M(Rect(0, 3, 4, 3)));
    
    // Fill point vectors
    M.at<double>(0, 4) = -x1[0];
    M.at<double>(1, 4) = -x1[1];
    M.at<double>(2, 4) = -x1[2];
    M.at<double>(3, 5) = -x2[0];
    M.at<double>(4, 5) = -x2[1];
    M.at<double>(5, 5) = -x2[2];
    
    Mat U, S, Vt;
    SVD::compute(M, S, U, Vt);
    
    // Get last row of V (transpose of last column of Vt)
    Mat V_last_col = Vt.row(Vt.rows - 1);
    Vec4d X;
    for (int i = 0; i < 4; i++) {
        X[i] = V_last_col.at<double>(i);
    }
    
    // Normalize
    return X / X[3];
}

ParticleSizeResult findParticleSizes(
    const vector<pair<string, string>>& selHour,
    const Mat& P1,
    const Mat& P0,
    double pre_x,
    double pre_y,
    const Mat& F10,
    const Vec3d& t0)
{
    vector<Point2f> coords1, coords0;
    vector<int> area_pixels;
    vector<double> diam_eq_mm;
    vector<string> minutes;
    vector<int> flakes;
    
    // Extract coordinates from strings
    for (const auto& f : selHour) {
        const auto& c1 = f.first;
        const auto& c0 = f.second;
        
        // Parse coordinates from strings
        // Assuming format contains "X123Y456r"
        auto getXY = [](const string& s) -> Point2f {
            size_t xPos = s.find('X') + 1;
            size_t yPos = s.find('Y') + 1;
            size_t rPos = s.rfind('r');
            int x = stoi(s.substr(xPos, s.find('Y') - xPos));
            int y = stoi(s.substr(yPos, rPos - 1 - yPos));
            return Point2f(static_cast<float>(x), static_cast<float>(y));
        };
        
        coords1.push_back(getXY(c1));
        coords0.push_back(getXY(c0));
    }
    
    // Correct matches using OpenCV
    vector<Point2f> nc1, nc0;
    correctMatches(F10, coords1, coords0, nc1, nc0);
    
    // Process each matched point
    for (size_t i = 0; i < nc1.size(); ++i) {
        // Convert to homogeneous coordinates
        Vec3d n1(nc1[i].x, nc1[i].y, 1.0);
        Vec3d n0(nc0[i].x, nc0[i].y, 1.0);
        
        // Triangulate point
        Vec4d p3d = triangulatePoint(n1, n0, P1, P0);
        
        // Calculate pixel area and dimensions
        double ppx = p3d[2] * pre_x;
        double ppy = p3d[2] * pre_y;
        
        // Read image and count non-zero pixels
        Mat img = imread(selHour[i].second, IMREAD_GRAYSCALE);
        int area_pixel = countNonZero(img);
        
        // Extract minute from filename
        string f = selHour[i].second;
        string minute = f.substr(
            findNth(f, "-", 4) + 1,
            findNth(f, "-", 5) - findNth(f, "-", 4) - 1
        );
        
        // Extract flake number from filename
        int flake = stoi(f.substr(
            findNth(f, "/", 4) + 6,
            6  // Assuming 6-digit flake numbers
        ));
        
        // Store results
        area_pixels.push_back(area_pixel);
        diam_eq_mm.push_back(sqrt(area_pixel * ppx * ppy * 4.0 / M_PI));
        minutes.push_back(minute);
        flakes.push_back(flake);
    }
    
    return make_tuple(area_pixels, diam_eq_mm, minutes, flakes);
}


// TODO: Bining
double calcSingleCamVolume() {
    return -1.0;
}

// TODO: Bining
void snowflakeBins() {}

#endif