#import sklearn.metrics import confusion_matrix
import numpy as np
import time
import os
from singleView import loadModel
from utils import getPrediction, getInput


path_to_weights = '/home/nvidia/SnowTeam/SnowTeam_AGX/weights/shape-0.95.hdf5'
path_to_image_folder = '/home/nvidia/SnowTeam/SnowTeam_AGX/imgs1/'
num_classes = 6
imageSize = 300


start_time = time.time()

model = loadModel(num_classes, path_to_weights, configFile='', imageSize=300, dropout=0.3, learningRate=0.0001)

images = os.listdir(path_to_image_folder)
print("found: " + str(len(images)) + " images in: " + path_to_image_folder)

# Uncomment one of these depending on the type of classification being done
outputDirs = [path_to_image_folder + "/small_particles", path_to_image_folder + "/columnar_crystals", path_to_image_folder + "/planar_crystals", path_to_image_folder + "/aggregrates", path_to_image_folder + "/graupel", path_to_image_folder + "/columnar_planar_combination"]
#outputDirs = [path_to_image_folder + "/unrimed", path_to_image_folder + "/rimed", path_to_image_folder + "/densely_rimed", path_to_image_folder + "/graupel_like", path_to_image_folder + "/graupel"]
#outputDirs = [path_to_image_folder + "/dry", path_to_image_folder + "/wet"]

for name in outputDirs:
    os.mkdir(name)
    print("created output directory: " + name)

print("Starting classification")

for flake in images:
    if ".png" in flake:
        img = getInput(path_to_image_folder + flake, imageSize=imageSize)
        pred = getPrediction(img, model)
        savePath = outputDirs[pred] + "/" + flake
        os.rename(path_to_image_folder + flake, savePath)
print("----execution took %s seconds----" % (time.time()-start_time)) 
