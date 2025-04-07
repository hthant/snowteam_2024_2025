#ifndef __MATCHINGCODE_H_INCLUDED__  
#define __MATCHINGCODE_H_INCLUDED__

#include <vector>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void matchPics(vector<filesystem::directory_entry> cam1Images, vector<filesystem::directory_entry> cam2Images, Mat fundamentalMatrix,
               string outputFile);

#endif