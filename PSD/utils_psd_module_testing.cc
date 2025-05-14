#ifndef __UTILS_PSD_MODULE_TESTING__
#define __UTILS_PSD_MODULE_TESTING__

#include "utils_psd.cc"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;
using namespace cv;

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
Mat F = (Mat_<double>(3,3) << 
    7.52298113e-08, 2.80363662e-07, -3.83911844e-04,
    2.18704253e-07, -4.17815657e-08, -1.50563382e-03,
    -2.35591486e-05, 1.84734377e-03, 1.00000000e+00);
vector<pair<string, string>> hr28;

void vectorPairPrint(vector<pair<string, string>> pairs) {
    for (const pair<string, string>& p : pairs) {
        cout << "[" << p.first << ", " << p.second << "]" << endl;
    }
}

template <typename T>
void printVector(const std::vector<T>& vec) {
    std::cout << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i];
        if (i < vec.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]";
}

// Function to print the ParticleSizeResult tuple
void printParticleSizeResult(const ParticleSizeResult& result) {
    std::cout << "ParticleSizeResult: \n";
    std::cout << "area_pixels: " << std::endl;
    printVector(std::get<0>(result));
    std::cout << "\ndiam_eq_mm: " << std::endl;
    printVector(std::get<1>(result));
    std::cout << "\nminutes: " << std::endl;
    printVector(std::get<2>(result));
    std::cout << "\nflakes: " << std::endl;
    printVector(std::get<3>(result));
    std::cout << std::endl;
}

void matchingInputReader(string input_file) {
    ifstream file(input_file);
    string line;

    if (file.is_open()) {
        while (getline(file, line)) {
            stringstream ss(line);
            string item1;
            string item2;

            // Read items separated by commas
            if (getline(ss, item1, ',') && getline(ss, item2, ',')) {
                hr28.push_back(make_pair(item1, item2)); // Store the pair
            }
        }
        file.close();
    } else {
        cerr << "Unable to open file." << endl;
    }
}

void testbench(int px_x, int px_y) {
    pair<Mat, Mat> R01_t01 = computeReverseRT(R0, t0);
    Mat R01 = R01_t01.first;
    Mat t01 = R01_t01.second;
    cout << R01 << endl;
    cout << t01 << endl;

    Mat P1 = computeProjection(A1, R01, t01);
    cout << P1 << endl;

    Mat R = Mat::eye(3, 3, CV_64F);
    Mat t = Mat::zeros(3, 1, CV_64F);
    Mat P0 = computeProjection(A0, R, t);
    cout << P0 << endl;

    double ang_x = findFov(px_x, A0.at<double>(0,0));
    double ang_y = findFov(px_y, A0.at<double>(1,1));
    cout << ang_x << endl;
    cout << ang_y << endl;
    
    pair<double,double> pre_xy = calculatePreConstants(ang_x, ang_y, px_x, px_y);
    double pre_x = pre_xy.first;
    double pre_y = pre_xy.second;
    cout << pre_x << endl;
    cout << pre_y << endl;

    ParticleSizeResult test = findParticleSizes(hr28, P1, P0, pre_x, pre_y, F, t0);
    printParticleSizeResult(test);
}

int main() {
    int px_x = 2448;
    int px_y = 2048;

    matchingInputReader("matching_input.txt");
    //vectorPairPrint(hr28);
    testbench(px_x, px_y);
}

#endif