#ifndef __UTILS_CC_INCLUDED__  
#define __UTILS_CC_INCLUDED__

#include "utils.h"
#include <vector>
#include <filesystem>
#include <algorithm>
#include <regex>
#include <utility>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

using namespace cv;
using namespace std;
using json = nlohmann::json;

// ================================================================================================
// Image Matching Utilities
// ================================================================================================

// Reads in all the files in a directory and writes them into a vector of directory_entry
vector<filesystem::directory_entry> vectorizeFileDirectory(string path){
    vector<filesystem::directory_entry> returnVec;
    for (const auto& entry : filesystem::directory_iterator(path)) {
        returnVec.push_back(entry);
    }
    return returnVec;
}

// No longer needed, sorts the vector of directory entry by date
void sortByDate(vector<filesystem::directory_entry>& images) {
    sort(images.begin(), images.end(),  
    [](const filesystem::directory_entry & a, const filesystem::directory_entry & b) {return a.last_write_time() < b.last_write_time();});
}

// compares two file pairs by the integer contained in the string of the pair.
bool compareFiles(pair<string, filesystem::directory_entry> &a, pair<string, filesystem::directory_entry> &b )
{
    return stoi(a.first) < stoi(b.first); 
}

// Takes two maps and return two maps only contianing the shared elemenets of both maps
pair<map<string, vector<filesystem::directory_entry>>, map<string, vector<filesystem::directory_entry>>> intersectMaps( 
    map<string, vector<filesystem::directory_entry>> map1, map<string, vector<filesystem::directory_entry>> map2) {
    map<string, vector<filesystem::directory_entry>> map1Result;
    map<string, vector<filesystem::directory_entry>> map2Result;
    for (const auto& [key, value] : map1) {
        if (map2.find(key) != map2.end()) {
            map1Result[key] = value;
            map2Result[key] = map2[key];
        }
    }
    return {map1Result, map2Result};
}

// returns a vector of points that contains the XY coordinates contained in the filename
vector<Point2f> toOutput(vector<filesystem::directory_entry> fileList) {
    vector<Point2f> vectorPairXY;
    for ( auto file : fileList) {
        string filename = file.path().string();
        vector<float> pairXY;
        regex rgx("X[0-9]*");
        regex rgy("Y[0-9]*");
        smatch matchX;
        smatch matchY;
        if(regex_search(filename, matchX, rgx)){
            string xCoordnate = matchX[0];
            pairXY.push_back(stof(xCoordnate.substr(1)));
        }
        if(regex_search(filename, matchY, rgy)){
            string yCoordnate = matchY[0];
            pairXY.push_back(stof(yCoordnate.substr(1)));
        }
        vectorPairXY.push_back(Point2f(pairXY[0], pairXY[1]));
    }
    return vectorPairXY;
}

// Finds the shortest distance between a point and a line
double shortestDist(Point2f point, Vec3f line) {
    float a = line[0];
    float b = line[1];
    float c = line[2];
    float xp = point.x;
    float yp = point.y;

    double x = (1 / (pow(a, 2) + pow(b, 2))) * ((pow(b, 2) * xp) - (a * b * yp) - (a * c));
    double y = ((-a * x) / b) - (c / b);
    
    return sqrt(pow((yp - y), 2) + pow((xp-x), 2));
}

// Uses the points vector of points and the fundamental matrix to calculate epilines, it then chooses the closest image if less than the distance
// threshold and records the indexes so they can be matched together in a later step
pair<vector<int>, vector<int>> findCorrespondIndicies(vector<Point2f> pts1, vector<Point2f> pts2, Mat fundamentalMatrix, pair<int, int> img1Size, pair<int, int> img2Size, 
                                   int distanceThresh) {
    vector<Vec3f> lines2; 
    vector<Vec3f> lines1; 
    computeCorrespondEpilines(pts1, 1, fundamentalMatrix, lines2);
    computeCorrespondEpilines(pts2, 2, fundamentalMatrix, lines1);

    vector<int> matchingIndicies2to1(pts1.size(), -99);
    vector<int> matchingIndicies1to2(pts2.size(), -99);
    double dist;
    for(size_t i = 0; i < lines2.size(); i ++) {
        for(size_t j = 0; j < pts2.size(); j++) {
            dist = shortestDist(pts2[j], lines2[i]);
            if(dist < distanceThresh) {
                matchingIndicies2to1[i] = j;
            }
        }
    }
    for(size_t i = 0; i < lines1.size(); i ++) {
        for(size_t j = 0; j < pts1.size(); j++) {
            dist = shortestDist(pts1[j], lines1[i]);
            if(dist < distanceThresh) {
                matchingIndicies1to2[i] = j;
            }
        }
    }
    return {matchingIndicies1to2, matchingIndicies2to1};
}


// ================================================================================================
// Fundamental Matrix Utilities
// ================================================================================================

// Takes a large file list and returns a map with the large vector divided into four smaller file list by name
map<string, vector<filesystem::directory_entry>> sortFileDirectory( vector<filesystem::directory_entry> fileList, 
                                                                    string regx1, string regx2, string regx3, string regx4) {
    map<string, vector<filesystem::directory_entry>> returnSorted;
    for(auto file : fileList){
        string filename = file.path().string();
        regex rgx1(regx1);
        regex rgx2(regx2);
        regex rgx3(regx3);
        regex rgx4(regx4);
        smatch match;
        if(regex_search(filename, match, rgx1)){
           returnSorted[regx1].push_back(file);
        }
        if(regex_search(filename, match, rgx2)){
           returnSorted[regx2].push_back(file);
        }
        if(regex_search(filename, match, rgx3)){
           returnSorted[regx3].push_back(file);
        }
        if(regex_search(filename, match, rgx4)){
           returnSorted[regx4].push_back(file);
        }
    }
    return returnSorted;
}


// Reads a given cv of float values and converts it into a vector of float vectors
vector<vector<float>> readCSV(string fileName){
    ifstream inputFile(fileName);
    if(!inputFile.is_open()) {
        cerr << "Could not open file: " << fileName << "\n";
        exit(1);
    }
    string line;
    vector<vector<float>> resultVector;
    while(getline(inputFile, line)) {
        stringstream ss(line);
        string item;
        vector<float> row;
        while (std::getline(ss, item, ',')) {
            row.push_back(stof(item));
        }
        resultVector.push_back(row);
    }
    return resultVector;
}

// Adds a padding element to a vector
vector<vector<float>> padVector(vector<vector<float>>& input, float padding) {
    for(auto& elem : input) {
        elem.push_back(padding);
    }
    return input;
}


// Converts a vector of float vectors into an openCV matrix of floats
Mat convertVectorToMat(vector<vector<float>> input) {
    Mat returnMat(int(input.size()), int(input[0].size()), CV_32F);
    for(size_t i = 0; i < input.size(); i++){
        for(size_t j = 0; j < input[j].size(); j++) {
            returnMat.at<float>(i, j) = input[i][(j+1) % 2];
        }
    }
    return returnMat;
}

// ================================================================================================
// Main Function Utilities
// ================================================================================================

// Reads the config json into a map
map<string,string> readConfigJson(string filename) {
    std::ifstream file(filename);
    json j;
    file >> j;

    map<string, string> configMap;

    for (json::iterator it = j.begin(); it != j.end(); ++it) {
        configMap[it.key()] = it.value();
    }

    return configMap;
    } 

// Reads the fundamental matrix json into a map of double vectors
map<string, vector<double>> readMatrixJson(string filename){
    std::ifstream file(filename);
    json j;
    file >> j;

    map<string, vector<double>> matrixMap;

    for (json::iterator it = j.begin(); it != j.end(); ++it) {
        vector<double> matrixVector;
        for (auto elem : it.value()) {
            matrixVector.push_back(elem);
        }
        matrixMap[it.key()] = matrixVector;
    }

    return matrixMap;
} 

// Writes a map of double vectors into a json
void writeJson(map<string, vector<double>> matrixMap){
    ofstream file("fundamentalMatrix.json");
    json j;
    for(const auto& [key, value] : matrixMap){
        j[key] = value;
    }
    file << j;
}

// converts an OpenCv matrix into a vector of doubles
vector<double> convertMatToVector(Mat matrix){
    vector<double> returnVec;
    for(size_t i = 0; i < floor(sqrt(matrix.total())); i++){
        for(size_t j = 0; j < floor(sqrt(matrix.total())); j++) {
            returnVec.push_back(matrix.at<double>(i, j));
        }
    }
    return returnVec;
}

// Conveerts a 1d vector of doubles into a matrix (specifically 3x3)
Mat convert1dVectorToMat(vector<double> input) {
    Mat returnMat(3, 3, CV_32F);
    int count = 0;
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++) {
            returnMat.at<float>(i, j) = input[count];
            count ++;
        }
    }
    return returnMat;
}

// Splits a string into a vector of strings based on the delimter
vector<string> split(const string &s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::stringstream ss(s);

    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }

    return tokens;
}

// Combines 3 files into one file if two pairs share an element, it makes a set of three elements all refering to the same snowflake,
// if not then it just adds the pair to the list
void combineFiles(vector<string> enablePair, vector<string> outputFilePair, vector<pair<int, int>> indexPair, string outFileName){
    vector<vector<vector<string>>> combinedFileData;
    for(size_t i = 0; i < enablePair.size(); i++) {
        if(enablePair[i] == "True"){
            ifstream file(outputFilePair[i]);
            if(file.is_open()){
                vector<vector<string>> individualFileData;
                string line;
                while (getline(file, line)) {
                    individualFileData.push_back(split(line, ','));
                }
                file.close();
                combinedFileData.push_back(individualFileData);
            }
        }
        else{
            combinedFileData.push_back({{}});
        }
    }

    vector<vector<string>> outFileData;

    for(size_t i = 0; i < combinedFileData.size(); i++) {
        if(enablePair[i] == "True"){
            for(auto row : combinedFileData[i]) {
                bool contains = false;
                for ( auto outRow : outFileData){
                    if(row[0] == outRow[indexPair[i].first]) {
                        outRow[indexPair[i].second] = row[1];
                        contains = true;
                        break;
                    }
                    if(row[1] == outRow[indexPair[i].second]) {
                        outRow[indexPair[i].first] = row[0];
                        contains = true;
                        break;
                    }
                }
                if(!contains) {
                    vector<string> newLine;
                    for(size_t j = 0; j < 7; j++){
                        if(indexPair[i].first == (int) j) {
                            newLine.push_back(row[0]);
                        }
                        else if(indexPair[i].second == (int) j) {
                            newLine.push_back(row[1]);
                        }
                        else{
                            newLine.push_back("NA");
                        }
                    }
                    outFileData.push_back(newLine);
                }
            }
        }
    }

    ofstream outFile(outFileName);
    if (outFile.is_open()) {
        for (auto row : outFileData) {
            outFile << row[0] << "," << row[1] << "," << row[2] << "," << row[3] << "," << row[4] << "," 
                    << row[5] << "," << row[6] << "\n";
        }
    }
    outFile.close();
}
 
// Takes the combined file list and all the individual file lists. If any file from the individual list are not included in the combined file list
// they are added to the output
void createFullList(vector<string> cameraPath, vector<string> cameraEnable, string combinedFileName, string outFileName){
    ifstream combinedFileRaw(combinedFileName);
    vector<vector<string>> combinedFile;
    string line;
    vector<string> combinedLine;
    vector<filesystem::directory_entry> camFiles;
    bool contains;
    vector<string> newLine;

    if (combinedFileRaw.is_open()) {
        while (getline(combinedFileRaw, line)) {
            combinedLine = split(line, ',');
            combinedFile.push_back(combinedLine);
        }
        combinedFileRaw.close();
    }

    for(size_t i = 0; i < cameraPath.size(); i++){
        if(cameraEnable[i] == "True"){
            camFiles = vectorizeFileDirectory(cameraPath[i]);
            for(const auto& photo : camFiles) {
                contains = false;
                for(size_t j = 0; j < combinedFile.size(); j ++){
                    if(photo.path().string() == combinedFile[j][i]){
                        contains = true;
                    }
                }
                if(!contains) {
                    vector<string> newLine;
                    for(size_t k = 0; k < 7; k++){ 
                        if(i == k) {
                            newLine.push_back(photo.path().string());
                        }
                        else{
                            newLine.push_back("NA");
                        }
                    }
                    combinedFile.push_back(newLine);
                }
            }
        }
    }

    ofstream outFile(outFileName);
    if (outFile.is_open()) {
        for (auto row : combinedFile) {
            outFile << row[0] << "," << row[1] << "," << row[2] << "," << row[3] << "," << row[4] << "," 
                    << row[5] << "," << row[6] << "\n";
        }
    }
    outFile.close();
}


#endif