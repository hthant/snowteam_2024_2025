#ifndef __FUNDAMENTALMATRIX_CC_INCLUDED__  
#define __FUNDAMENTALMATRIX_CC_INCLUDED__

#include "fundamentalMatrix.h"
#include "utils.h"
#include <filesystem>
#include <vector>
#include <sys/stat.h>
#include <opencv2/opencv.hpp>



using namespace std;
using namespace cv;

Mat calculateFundementalMatrix(string filePath, string cam0wPt, string cam0iPt, string cam1wPt, string cam1iPt) {    
    vector<vector<float>> imgPoints0_sel;
    vector<vector<float>> imgPoints1_sel;

    // Read in all the files, and sort them in order into their correct vectors.

    vector<filesystem::directory_entry> fileList;
    fileList = vectorizeFileDirectory(filePath);
    map<string, vector<filesystem::directory_entry>> sortedFiles;
    sortedFiles = sortFileDirectory(fileList, cam0wPt, cam0iPt, cam1wPt, cam1iPt);
    sort(sortedFiles[cam0wPt].begin(), sortedFiles[cam0wPt].end());
    sort(sortedFiles[cam0iPt].begin(), sortedFiles[cam0iPt].end());
    sort(sortedFiles[cam1wPt].begin(), sortedFiles[cam1wPt].end());
    sort(sortedFiles[cam1iPt].begin(), sortedFiles[cam1iPt].end());

    // cout << sortedFiles[cam0wPt].size() << " " << sortedFiles[cam0iPt].size() << " " << 
    //         sortedFiles[cam1wPt].size() << " " <<  sortedFiles[cam1iPt].size() << "\n";
    for(size_t i = 0; i < sortedFiles[cam0wPt].size() && i < sortedFiles[cam1wPt].size(); i++) {

        // Each files buffer size is big enough (I was given the numbers by Hein IDK why they are why they are) proceed. 

        struct stat buf1;
        struct stat buf2;
        struct stat buf3;
        struct stat buf4;

        stat(sortedFiles[cam0wPt][i].path().string().c_str(), &buf1);
        stat(sortedFiles[cam0iPt][i].path().string().c_str(), &buf2);
        stat(sortedFiles[cam1wPt][i].path().string().c_str(), &buf3);
        stat(sortedFiles[cam1iPt][i].path().string().c_str(), &buf4);
        if(buf1.st_size > 20 && buf2.st_size > 90 && buf3.st_size > 20 && buf4.st_size > 90) {

            // Convert the CSV files into vectors of float vectors and pad the vectors with 0's

            vector<vector<float>> w_Pt0 = readCSV(sortedFiles[cam0wPt][i].path().string());
            vector<vector<float>> i_Pt0 = readCSV(sortedFiles[cam0iPt][i].path().string());
            w_Pt0 = padVector(w_Pt0, 0);

            vector<vector<float>> w_Pt1 = readCSV(sortedFiles[cam1wPt][i].path().string());
            vector<vector<float>> i_Pt1 = readCSV(sortedFiles[cam1iPt][i].path().string());
            w_Pt1 = padVector(w_Pt1, 0);

            // Find the intersections of w_Pt0 and wPt1 and store in the intersection vector 

            vector<vector<float>> intersection;
            sort(w_Pt0.begin(), w_Pt0.end());
            sort(w_Pt1.begin(), w_Pt1.end());
            set_intersection(w_Pt0.begin(), w_Pt0.end(), w_Pt1.begin(), w_Pt1.end(), std::back_inserter(intersection));
            
            // For each intersection find the index of each intersection row and push back the value into indA/indB
            // Then make a vector of values from the same rows from w_Pt but using the i_Pt vectors

            vector<size_t> indA;
            vector<size_t> indB;
            for(auto row : intersection) {
                indA.push_back(distance(w_Pt0.begin(), find(w_Pt0.begin(), w_Pt0.end(), row)));
                indB.push_back(distance(w_Pt1.begin(), find(w_Pt1.begin(), w_Pt1.end(), row)));
            }
            for(size_t j = 0; j < indA.size(); j++) {
                imgPoints0_sel.push_back(i_Pt0[indA[j]]);
                imgPoints1_sel.push_back(i_Pt1[indB[j]]);
            }
        
        }
            
    }
    
    // Convert those vector of float vectors into openCv float Matrix

    Mat c0points = convertVectorToMat(imgPoints0_sel);
    Mat c1points = convertVectorToMat(imgPoints1_sel);

    // Calculate the fundamental matrix using openCV and return it

    Mat F = findFundamentalMat(c0points, c1points, cv::FM_RANSAC, 0.05, 0.999);
    return F;
}

#endif