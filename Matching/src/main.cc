#ifndef __MAIN_CC_INCLUDED__  
#define __MAIN_CC_INCLUDED__

#include "matchingCode.h"
#include "utils.h"
#include "fundamentalMatrix.h"
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace chrono;

// Reads config json and readMatix json into maps to be read later
// For each camera pair if enable, either calcualate the fundamental matrix or read the stored fundamental matrix
// Then create a vector of all the images in the correct folder
// Match all the pictures between the two camera and place the list of the matches in the correct file.

// After doing that for all camera pairs, if enabled, put all pairs in the same file so its convient.
// Additionally if enabled put all pictures, paired or not in the same file

int main() {
    auto start = high_resolution_clock::now();
    map<string, string> configMap = readConfigJson("config.json");
    map<string, vector<double>> matrixMap = readMatrixJson("fundamentalMatrix.json");
    vector<pair<int, int>> indexPair = {{0,1}, {0,3}, {0,6}, {1,2}, {1,3}, {1,4}, {2,4}, {2,5}, {3,4}, {3,6}, {4,5}, {5,6}};  
    vector<string> matrixNames = {"F01", "F03", "F06", "F12", "F13", "F14", "F24", "F25", "F34", "F36", "F45", "F56"};
    vector<string> enablePair = {configMap["enable01"], configMap["enable03"], configMap["enable06"], configMap["enable12"], configMap["enable13"], configMap["enable14"],
                                 configMap["enable24"], configMap["enable25"], configMap["enable34"], configMap["enable36"], configMap["enable45"], configMap["enable56"]};
    vector<string> calculateMatrixPair = {configMap["calculateMatrix01"], configMap["calculateMatrix03"], configMap["calculateMatrix06"], configMap["calculateMatrix12"], configMap["calculateMatrix13"], configMap["calculateMatrix14"],
                                          configMap["calculateMatrix24"], configMap["calculateMatrix25"], configMap["calculateMatrix34"], configMap["calculateMatrix36"], configMap["calculateMatrix45"], configMap["calculateMatrix56"]};
    vector<string> fundamentalMatrixPair = {configMap["fundamentalMatrix01"], configMap["fundamentalMatrix03"], configMap["fundamentalMatrix06"], configMap["fundamentalMatrix12"], configMap["fundamentalMatrix13"], configMap["fundamentalMatrix14"],
                                            configMap["fundamentalMatrix24"], configMap["fundamentalMatrix25"], configMap["fundamentalMatrix34"], configMap["fundamentalMatrix36"], configMap["fundamentalMatrix45"], configMap["fundamentalMatrix56"]};
    vector<string> fundamentalMatrixRegx0Pair  = {configMap["fundamentalMatrix01Regx0"], configMap["fundamentalMatrix03Regx0"], configMap["fundamentalMatrix06Regx0"], configMap["fundamentalMatrix12Regx0"], 
                                                  configMap["fundamentalMatrix13Regx0"], configMap["fundamentalMatrix14Regx0"], configMap["fundamentalMatrix24Regx0"], configMap["fundamentalMatrix25Regx0"], 
                                                  configMap["fundamentalMatrix34Regx0"], configMap["fundamentalMatrix36Regx0"], configMap["fundamentalMatrix45Regx0"], configMap["fundamentalMatrix56Regx0"]};
    vector<string> fundamentalMatrixRegx1Pair  = {configMap["fundamentalMatrix01Regx1"], configMap["fundamentalMatrix03Regx1"], configMap["fundamentalMatrix06Regx1"], configMap["fundamentalMatrix12Regx1"], 
                                                  configMap["fundamentalMatrix13Regx1"], configMap["fundamentalMatrix14Regx1"], configMap["fundamentalMatrix24Regx1"], configMap["fundamentalMatrix25Regx1"], 
                                                  configMap["fundamentalMatrix34Regx1"], configMap["fundamentalMatrix36Regx1"], configMap["fundamentalMatrix45Regx1"], configMap["fundamentalMatrix56Regx1"]};
    vector<string> fundamentalMatrixRegx2Pair  = {configMap["fundamentalMatrix01Regx2"], configMap["fundamentalMatrix03Regx2"], configMap["fundamentalMatrix06Regx2"], configMap["fundamentalMatrix12Regx2"], 
                                                  configMap["fundamentalMatrix13Regx2"], configMap["fundamentalMatrix14Regx2"], configMap["fundamentalMatrix24Regx2"], configMap["fundamentalMatrix25Regx2"], 
                                                  configMap["fundamentalMatrix34Regx2"], configMap["fundamentalMatrix36Regx2"], configMap["fundamentalMatrix45Regx2"], configMap["fundamentalMatrix56Regx2"]};
    vector<string> fundamentalMatrixRegx3Pair  = {configMap["fundamentalMatrix01Regx3"], configMap["fundamentalMatrix03Regx3"], configMap["fundamentalMatrix06Regx3"], configMap["fundamentalMatrix12Regx3"], 
                                                  configMap["fundamentalMatrix13Regx3"], configMap["fundamentalMatrix14Regx3"], configMap["fundamentalMatrix24Regx3"], configMap["fundamentalMatrix25Regx3"], 
                                                  configMap["fundamentalMatrix34Regx3"], configMap["fundamentalMatrix36Regx3"], configMap["fundamentalMatrix45Regx3"], configMap["fundamentalMatrix56Regx3"]};
    vector<string> cameraPath = {configMap["cam0Path"], configMap["cam1Path"], configMap["cam2Path"], configMap["cam3Path"], configMap["cam4Path"], configMap["cam5Path"], configMap["cam6Path"]};
    vector<string> cameraEnable = {configMap["cam0Enable"], configMap["cam1Enable"], configMap["cam2Enable"], configMap["cam3Enable"], configMap["cam4Enable"], configMap["cam5Enable"], configMap["cam6Enable"]};
    vector<string> outputFilePair = {configMap["outputFile01"], configMap["outputFile03"], configMap["outputFile06"], configMap["outputFile12"], configMap["outputFile13"], configMap["outputFile14"],
                                     configMap["outputFile24"], configMap["outputFile25"], configMap["outputFile34"], configMap["outputFile36"], configMap["outputFile45"], configMap["outputFile56"]};
    vector<vector<double>> FValues = {matrixMap["F01"], matrixMap["F03"], matrixMap["F06"], matrixMap["F12"], matrixMap["F13"], matrixMap["F14"], 
                                      matrixMap["F24"], matrixMap["F25"], matrixMap["F34"], matrixMap["F36"], matrixMap["F45"], matrixMap["F56"]};
    try {
        for(size_t i = 0; i < enablePair.size(); i++){
            if(enablePair[i] == "True"){
                Mat F;
                if(calculateMatrixPair[i] == "True") {
                    F = calculateFundementalMatrix(fundamentalMatrixPair[i], fundamentalMatrixRegx0Pair[i], fundamentalMatrixRegx1Pair[i], fundamentalMatrixRegx2Pair[i], fundamentalMatrixRegx3Pair[i]);
                    FValues[i] = (convertMatToVector(F));
                    matrixMap[matrixNames[i]] = FValues[i];

                }
                F = convert1dVectorToMat(FValues[i]);
                vector<filesystem::directory_entry> camAImages = vectorizeFileDirectory(cameraPath[indexPair[i].first]);
                vector<filesystem::directory_entry> camBImages = vectorizeFileDirectory(cameraPath[indexPair[i].second]);    
                matchPics(camAImages, camBImages, F, outputFilePair[i]);                         
            }
        }
    } catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation failed at 01: " << e.what() << '\n';
        exit(1);
    }

    combineFiles(enablePair, outputFilePair, indexPair, configMap["combinedOutputFile"]);

    if(configMap["createFullList"] == "True") {
        createFullList(cameraPath, cameraEnable, configMap["combinedOutputFile"], configMap["fullOutputFile"]);
    }

    writeJson(matrixMap);

    auto stop = high_resolution_clock::now();
    cout << "----execution took " << chrono::duration_cast<milliseconds>(stop - start).count() << " Milli Seconds----\n";

}

#endif 