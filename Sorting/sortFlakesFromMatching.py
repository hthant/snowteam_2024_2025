#import sklearn.metrics import confusion_matrix
import numpy as np
import time
import os
import csv
import re
import singleView
import multiView
from utils import getPrediction, getInput, getInput3Channels, getPrediction

path_to_weights = '/home/nvidia/SnowTeam/SnowTeam_AGX/weights/shape-0.95.hdf5'
output_file_path = '/home/nvidia/SnowTeam/SnowTeam_AGX/snowImageProcessing/matching/output/fullOutputFile.txt'
matching_base_path = '/home/nvidia/SnowTeam/SnowTeam_AGX/snowImageProcessing/matching'
num_classes = 6
imageSize = 300


start_time = time.time()

singleViewModel = singleView.loadModel(num_classes, path_to_weights, configFile='', imageSize=300, dropout=0.3, learningRate=0.0001)
multiViewModel = multiView.loadModel(num_classes, path_to_weights, configFile='', imageSize=300, dropout=0.3, learningRate=0.0001)

# Uncomment one of these depending on the type of classification being done
outputDirs = [matching_base_path + "/small_particles", matching_base_path + "/columnar_crystals", matching_base_path + "/planar_crystals", matching_base_path + "/aggregrates", matching_base_path + "/graupel", matching_base_path + "/columnar_planar_combination"]
#outputDirs = [matching_base_path + "/unrimed", matching_base_path + "/rimed", matching_base_path + "/densely_rimed", matching_base_path + "/graupel_like", matching_base_path + "/graupel"]
#outputDirs = [matching_base_path + "/dry", matching_base_path + "/wet"]

for name in outputDirs:
    os.mkdir(name)
    print("created output directory: " + name)

print("Starting classification")

import csv

with open(output_file_path, 'r') as file:
    csvreader = csv.reader(file)

    rowCount = 0    
    for row in csvreader:
        images = re.split(",",row)
        imageCount = 0
        realImages = []
        for imageName in images:
            if imageName != "NA":
                imageCount += 1
                realImages.__add__(imageName)
        if imageCount == 1:
            for flake in realImages:
                img = getInput(matching_base_path + flake, imageSize=imageSize)
                pred = getPrediction(img, singleViewModel)
                savePath = outputDirs[pred] + "/" + flake
                os.rename(matching_base_path + flake, savePath)
        if imageCount == 3:
            imgs = getInput3Channels(flake, imageSize=imageSize)
            pred = getPrediction(imgs, multiViewModel)
            savePath = outputDirs[pred] + '/flake' + str(rowCount)
            os.mkdir(savePath)
            os.rename(flake[0], savePath + '/' + flake[0][flake[0].index('cropped/')+8:])
            os.rename(flake[1], savePath + '/' + flake[1][flake[1].index('cropped/')+8:])
            os.rename(flake[2], savePath + '/' + flake[2][flake[2].index('cropped/')+8:])
        rowCount += 1

print("----execution took %s seconds----" % (time.time()-start_time)) 
