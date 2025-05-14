import time
import os, sys
from multiView import loadModel
from utils import getInput3Channels, getPrediction, getList


path_to_weights = '/home/snow/multiview/classification/melting/weights/fineTuning/weights-improvement-08-0.91.hdf5'
path_to_image_folder = '/home/snow/multiview/images/melting'
num_classes = 2
imageSize = 300

# Paths to matching outputs
path1 = '/home/snow/multiview/matching/pairedFlakeKey24.txt'
path2 = '/home/snow/multiview/matching/pairedFlakeKey25.txt'


start_time = time.time()

model = loadModel(num_classes, path_to_weights, configFile='', imageSize=300, dropout=0.3, learningRate=0.0001)

flakes = getList(path1, path2, v=0)
if len(flakes) == 0:
    print("\n\n\tWARNING: No flake triplets found given matching output\n\tTERMINATING")
    sys.exit(0)

# Uncomment one of these depending on the type of classification being done
#outputDirs = [path_to_image_folder + "/small_particles", path_to_image_folder + "/columnar_crystals", path_to_image_folder + "/planar_crystals", path_to_image_folder + "/aggregrates", path_to_image_folder + "/graupel", path_to_image_folder + "/columnar_planar_combination"]
#outputDirs = [path_to_image_folder + "/unrimed", path_to_image_folder + "/rimed", path_to_image_folder + "/densely_rimed", path_to_image_folder + "/graupel_like", path_to_image_folder + "/graupel"]
outputDirs = [path_to_image_folder + "/dry", path_to_image_folder + "/wet"]

for name in outputDirs:
    os.mkdir(name)
    print("created output directory: " + name)

print("Starting classification")

for i, flake in enumerate(flakes):
        imgs = getInput3Channels(flake, imageSize=imageSize)
        pred = getPrediction(imgs, model)
        savePath = outputDirs[pred] + '/flake' + str(i)
        os.mkdir(savePath)
        os.rename(flake[0], savePath + '/' + flake[0][flake[0].index('cropped/')+8:])
        os.rename(flake[1], savePath + '/' + flake[1][flake[1].index('cropped/')+8:])
        os.rename(flake[2], savePath + '/' + flake[2][flake[2].index('cropped/')+8:])
print("----execution took %s seconds----" % (time.time()-start_time)) 
