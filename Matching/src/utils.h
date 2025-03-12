#ifndef __UTILS_H_INCLUDED__  
#define __UTILS_H_INCLUDED__

#include <vector>
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// ================================================================================================
// Image Matching Utilities
// ================================================================================================

vector<filesystem::directory_entry> vectorizeFileDirectory(string path);

void sortByDate(vector<filesystem::directory_entry>& images);

bool compareFiles(pair<string, filesystem::directory_entry> &a, pair<string, filesystem::directory_entry> &b );

pair<map<string, vector<filesystem::directory_entry>>, map<string, vector<filesystem::directory_entry>>> intersectMaps(
    map<string, vector<filesystem::directory_entry>> map1, map<string, vector<filesystem::directory_entry>> map2);

vector<Point2f> toOutput(vector<filesystem::directory_entry> fileList);

double shortestDist(Point2f point, Vec3f line);

pair<vector<int>, vector<int>> findCorrespondIndicies(vector<Point2f> pts1, vector<Point2f> pts2, Mat fundamentalMatrix,
                                                     pair<int, int> img1Size, pair<int, int> img2Size, int distanceThresh = 15);

// ================================================================================================
// Fundamental Matrix Utilities
// ================================================================================================

map<string, vector<filesystem::directory_entry>> sortFileDirectory( vector<filesystem::directory_entry> fileList, 
                                                                    string regx1, string regx2, string regx3, string regx4);

vector<vector<float>> readCSV(string fileName);

vector<vector<float>> padVector(vector<vector<float>>& input, float padding);

Mat convertVectorToMat(vector<vector<float>> input);

// ================================================================================================
// Main Function Utilities
// ================================================================================================

map<string, string> readConfigJson(string filename);

map<string, vector<double>> readMatrixJson(string filename);

void writeJson(map<string, vector<double>> matrixMap);

vector<double> convertMatToVector(Mat matrix);

Mat convert1dVectorToMat(vector<double> input);

vector<string> split(const string &s, char delimiter);

void combineFiles(string filename1, string filename2, string filename3, string outFileName);

void createFullList(string filePath0, string filePath1, string filePath2, string combinedFileName, string outFileName);

#endif