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

/* First Attempt
int main() {
    auto start = high_resolution_clock::now();
    map<string, string> configMap = readConfigJson("config.json");
    map<string, vector<double>> matrixMap = readMatrixJson("fundamentalMatrix.json");

    // Fundemental Matrix for the two camera being matched
    try {
        if(configMap["enable01"] == "True"){
            Mat F01;
            if(configMap["calculateMatrix01"] == "True") {
                F01 = calculateFundementalMatrix(configMap["fundamentalMatrix01"], configMap["fundamentalMatrix01Regx0"], 
                                                     configMap["fundamentalMatrix01Regx1"], configMap["fundamentalMatrix01Regx2"], 
                                                     configMap["fundamentalMatrix01Regx3"]);
                matrixMap["F01"] = convertMatToVector(F01);
            }
            else {
                F01 = convert1dVectorToMat(matrixMap["F01"]);
            }
        vector<filesystem::directory_entry> cam0Images;
        vector<filesystem::directory_entry> cam1Images;                             
        cam0Images = vectorizeFileDirectory(configMap["cam0Path"]);
        cam1Images = vectorizeFileDirectory(configMap["cam1Path"]);
        matchPics(cam0Images, cam1Images, F01, configMap["outputFile01"]);
        }
    } catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation failed at 01: " << e.what() << '\n';
        exit(1);
    }

    try {
        if(configMap["enable02"] == "True"){
            Mat F02;
            if(configMap["calculateMatrix02"] == "True") {
                F02 = calculateFundementalMatrix(configMap["fundamentalMatrix02"], configMap["fundamentalMatrix02Regx0"], 
                                                     configMap["fundamentalMatrix02Regx1"], configMap["fundamentalMatrix02Regx2"], 
                                                     configMap["fundamentalMatrix02Regx3"]);
                matrixMap["F02"] = convertMatToVector(F02);
            }
            else{
                F02 = convert1dVectorToMat(matrixMap["F02"]);
            }
            vector<filesystem::directory_entry> cam0Images;
            vector<filesystem::directory_entry> cam2Images;
            cam0Images = vectorizeFileDirectory(configMap["cam0Path"]);
            cam2Images = vectorizeFileDirectory(configMap["cam2Path"]);
            matchPics(cam0Images, cam2Images, F02, configMap["outputFile02"]);
        }
    } catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation failed at 02: " << e.what() << '\n';
        exit(1);
    }

    try {
        if(configMap["enable12"] == "True"){
            Mat F12;
            if(configMap["calculateMatrix12"] == "True") {
                F12 = calculateFundementalMatrix(configMap["fundamentalMatrix12"], configMap["fundamentalMatrix12Regx0"], 
                                                     configMap["fundamentalMatrix12Regx1"], configMap["fundamentalMatrix12Regx2"], 
                                                     configMap["fundamentalMatrix12Regx3"]);
                matrixMap["F12"] = convertMatToVector(F12);
            }
            else{
                F12 = convert1dVectorToMat(matrixMap["F12"]);
            }
            vector<filesystem::directory_entry> cam1Images;
            vector<filesystem::directory_entry> cam2Images;
            cam1Images = vectorizeFileDirectory(configMap["cam1Path"]);
            cam2Images = vectorizeFileDirectory(configMap["cam2Path"]);
            matchPics(cam1Images, cam2Images, F12, configMap["outputFile12"]);
        }
    } catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation failed at 12: " << e.what() << '\n';
        exit(1);
    }

    try {
        if(configMap["enable34"] == "True"){
            Mat F34;
            if(configMap["calculateMatrix34"] == "True") {
                F34 = calculateFundementalMatrix(configMap["fundamentalMatrix34"], configMap["fundamentalMatrix34Regx0"], 
                                                     configMap["fundamentalMatrix34Regx1"], configMap["fundamentalMatrix34Regx2"], 
                                                     configMap["fundamentalMatrix34Regx3"]);
                matrixMap["F34"] = convertMatToVector(F34);
            }
            else{
                F34 = convert1dVectorToMat(matrixMap["F34"]);
            }

            vector<filesystem::directory_entry> cam3Images;
            vector<filesystem::directory_entry> cam4Images;
            cam3Images = vectorizeFileDirectory(configMap["cam3Path"]);
            cam4Images = vectorizeFileDirectory(configMap["cam4Path"]);
            matchPics(cam3Images, cam4Images, F34, configMap["outputFile34"]);
        }
    } catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation failed at 34: " << e.what() << '\n';
        exit(1);
    }

    try {
        if(configMap["enable35"] == "True"){
            Mat F35;
            if(configMap["calculateMatrix35"] == "True") {
                F35 = calculateFundementalMatrix(configMap["fundamentalMatrix35"], configMap["fundamentalMatrix35Regx0"], 
                                                     configMap["fundamentalMatrix35Regx1"], configMap["fundamentalMatrix35Regx2"], 
                                                     configMap["fundamentalMatrix35Regx3"]);
                matrixMap["F35"] = convertMatToVector(F35);
            }
            else{
                F35 = convert1dVectorToMat(matrixMap["F35"]);
            }

            vector<filesystem::directory_entry> cam3Images;
            vector<filesystem::directory_entry> cam5Images;
            cam3Images = vectorizeFileDirectory(configMap["cam3Path"]);
            cam5Images = vectorizeFileDirectory(configMap["cam5Path"]);
            matchPics(cam3Images, cam5Images, F35, configMap["outputFile35"]);
        }
    } catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation failed at 35: " << e.what() << '\n';
        exit(1);
    }

    try {
        if(configMap["enable45"] == "True"){
            Mat F45;
            if(configMap["calculateMatrix45"] == "True") {
                F45 = calculateFundementalMatrix(configMap["fundamentalMatrix45"], configMap["fundamentalMatrix45Regx0"], 
                                                     configMap["fundamentalMatrix45Regx1"], configMap["fundamentalMatrix45Regx2"], 
                                                     configMap["fundamentalMatrix45Regx3"]);
                matrixMap["F45"] = convertMatToVector(F45);
            }
            else{
                F45 = convert1dVectorToMat(matrixMap["F45"]);
            }

            vector<filesystem::directory_entry> cam4Images;
            vector<filesystem::directory_entry> cam5Images;
            cam4Images = vectorizeFileDirectory(configMap["cam0Path"]);
            cam5Images = vectorizeFileDirectory(configMap["cam2Path"]);
            matchPics(cam4Images, cam5Images, F45, configMap["outputFile45"]);
        }
    } catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation failed at 45: " << e.what() << '\n';
        exit(1);
    }

    if (configMap["enable01"] == "True" || configMap["enable02"] == "True" || configMap["enable12"] == "True") {
        combineFiles(configMap["outputFile01"], configMap["outputFile02"], configMap["outputFile12"], configMap["combinedOutputFile1"]);
    }

    if (configMap["enable34"] == "True" || configMap["enable35"] == "True" || configMap["enable45"] == "True") {
        combineFiles(configMap["outputFile34"], configMap["outputFile35"], configMap["outputFile45"], configMap["combinedOutputFile2"]);
    }

    if (configMap["createFullList1"] == "True"){
        createFullList(configMap["cam0Path"], configMap["cam1Path"], configMap["cam2Path"], configMap["combinedOutputFile1"], configMap["fullOutputFile1"]);
    }
    if (configMap["createFullList2"] == "True"){
        createFullList(configMap["cam3Path"], configMap["cam4Path"], configMap["cam5Path"], configMap["combinedOutputFile2"], configMap["fullOutputFile2"]);
    }

    writeJson(matrixMap);

    auto stop = high_resolution_clock::now();
    cout << "----execution took " << chrono::duration_cast<milliseconds>(stop - start).count() << " Milli Seconds----\n";

}
*/

int main() {
    cout << "1\n" << flush;
    auto start = high_resolution_clock::now();
    map<string, string> configMap = readConfigJson("config.json");
    map<string, vector<double>> matrixMap = readMatrixJson("fundamentalMatrix.json");
    cout << "2\n" << flush;
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
    vector<string> outputFilePair = {configMap["outputFile01"], configMap["outputFile03"], configMap["outputFile06"], configMap["outputFile12"], configMap["outputFile13"], configMap["outputFile14"],
                                     configMap["outputFile24"], configMap["outputFile25"], configMap["outputFile34"], configMap["outputFile36"], configMap["outputFile45"], configMap["outputFile56"]};
    vector<vector<double>> FValues = {matrixMap["F01"], matrixMap["F03"], matrixMap["F06"], matrixMap["F12"], matrixMap["F13"], matrixMap["F14"], 
                                      matrixMap["F24"], matrixMap["F25"], matrixMap["F34"], matrixMap["F36"], matrixMap["F45"], matrixMap["F56"]};
    cout << "3\n" << flush;
    try {
        for(size_t i = 0; i < enablePair.size(); i++){
            cout << "4\n" << flush; 
            if(enablePair[i] == "True"){
                cout << "5\n" << flush;
                Mat F;
                if(calculateMatrixPair[i] == "True") {
                    F = calculateFundementalMatrix(fundamentalMatrixPair[i], fundamentalMatrixRegx0Pair[i], fundamentalMatrixRegx1Pair[i], fundamentalMatrixRegx2Pair[i], fundamentalMatrixRegx3Pair[i]);
                    FValues[i] = (convertMatToVector(F));
                    cout << "Printing Vector\n" << flush;
                    for(auto elem : FValues[i]){
                        cout << elem << " ";
                    }
                    matrixMap[matrixNames[i]] = FValues[i];

                }
                cout << "6\n" << flush;
                F = convert1dVectorToMat(FValues[i]);
                cout << "7\n" << flush;
                vector<filesystem::directory_entry> camAImages = vectorizeFileDirectory(cameraPath[indexPair[i].first]);
                vector<filesystem::directory_entry> camBImages = vectorizeFileDirectory(cameraPath[indexPair[i].second]);    
                cout << "8\n" << flush;
                //matchPics(camAImages, camBImages, F, outputFilePair[i]);                         
            }
        }
    } catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation failed at 01: " << e.what() << '\n';
        exit(1);
    }
    cout << "9\n" << flush;

    // if (configMap["enable01"] == "True" || configMap["enable02"] == "True" || configMap["enable12"] == "True") {
    //     combineFiles(configMap["outputFile01"], configMap["outputFile02"], configMap["outputFile12"], configMap["combinedOutputFile1"]);
    // }

    // if (configMap["enable34"] == "True" || configMap["enable35"] == "True" || configMap["enable45"] == "True") {
    //     combineFiles(configMap["outputFile34"], configMap["outputFile35"], configMap["outputFile45"], configMap["combinedOutputFile2"]);
    // }

    // if (configMap["createFullList1"] == "True"){
    //     createFullList(configMap["cam0Path"], configMap["cam1Path"], configMap["cam2Path"], configMap["combinedOutputFile1"], configMap["fullOutputFile1"]);
    // }
    // if (configMap["createFullList2"] == "True"){
    //     createFullList(configMap["cam3Path"], configMap["cam4Path"], configMap["cam5Path"], configMap["combinedOutputFile2"], configMap["fullOutputFile2"]);
    // }

    writeJson(matrixMap);

    auto stop = high_resolution_clock::now();
    cout << "----execution took " << chrono::duration_cast<milliseconds>(stop - start).count() << " Milli Seconds----\n";

}

#endif 