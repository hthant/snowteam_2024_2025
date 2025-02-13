import os, sys
import numpy as np
import pandas as pd
import xarray
import dask
import random
import cv2 as cv
from sklearn.utils import shuffle
from PIL import Image
import mascdb.api
from mascdb.api import MASC_DB
from bisect import bisect
from scipy.ndimage import rotate
from typing import Any, Callable, Optional, Tuple
from sklearn.decomposition import PCA
import time
from concurrent.futures import ProcessPoolExecutor

CLASSNAMES = ['small_particle', 'columnar_crystal', 'planar_crystal', 'aggregrate', 'graupel', 'columnar_planar_combination']
RIMINGNAMES = ['unrimed', 'rimed', 'densely_rimed', 'graupel-like', 'graupel']

# =====================================================================================================
# GENERAL USE FUNCTIONS
# =====================================================================================================

#Load the mascDB data as an xarray dataset
def loadData(mascdbPath):
    print("Loading Dataset")
    mascdb = MASC_DB(dir_path=mascdbPath)
    return mascdb

#Save the Keras summary of a model
def saveModel(model, filePath):
    with open(filePath + ".txt",'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
    print("Model saved to: " + filePath + ".txt")

#Plot the accuracy and loss of a model's training from log file
def plotFromLog(filepath, title="Model Accuracy and Loss vs Nmber of Epochs trained for", plotAcc=True, plotLoss=True, plotValAcc=True, plotValLoss=True):
    acc = []
    loss = []
    val_acc = []
    val_loss = []
    f = open(filepath, 'r')

    for line in f.readlines():
        if ' acc: ' in line and ' val_acc: ' in line:
            acc.append(float(line[line.index('- acc: ')+7:line.index('- acc: ')+14]))
            loss.append(float(line[line.index('- loss: ')+7:line.index('- loss: ')+14]))
            val_acc.append(float(line[line.index(' val_acc: ')+9:line.index(' val_acc: ')+16]))
            val_loss.append(float(line[line.index(' val_loss: ')+10:line.index(' val_loss: ')+17]))

    f.close()
    epochs = np.arange(1, len(acc)+1).tolist()

    if plotAcc: plt.plot(epochs, acc, label='acc')
    if plotLoss: plt.plot(epochs, loss, label='loss')
    if plotValAcc: plt.plot(epochs, val_acc, label='val_acc')
    if plotValLoss: plt.plot(epochs, val_loss, label='val_loss')

    plt.xlabel("Epochs Trained For")
    plt.ylabel("Accuracy / Loss")
    plt.title("InceptionV3 With Pretrained Weights, Short Epochs, LR=0.001")
    #plt.yscale("log")
    #plt.ylim([0.15, 0.85])
    plt.legend(loc='upper left')

    plt.show()
    return plt

def newImage(img):
    arr = np.asarray(img)
    flip = random.randint(0, 3)
    rot = random.randint(0, 360)
    if flip >= 2:
        arr = np.flip(arr, 0)
    if flip > 0 and flip < 3:
        arr = np.flip(arr, 1)
    arr = rotate(arr, rot, reshape = False)
    noise = np.random.normal(0, 20, arr.shape)
    noised_image = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = xarray.DataArray(noised_image, img.coords, dims=img.dims)
    return img

#Save example images from data array
def saveImages(images, num, path):
    for i in range(num):
        filepath = path + str(i) + ".png"
        Image.fromarray(np.asarray(images[num])).save(filepath)


# =====================================================================================================
# SHAPE CLASSIFICATION
# =====================================================================================================

#Make an array of labels
def getLabels(numPerClass):
    labels = np.ones(numPerClass*6)
    n = 0
    for i in range(len(labels)):
        if i % (numPerClass) == 0:
            n += 1
        labels[i] = n
    return xarray.DataArray(labels, dims=["l"])

#Display information about how many snowflakes from each class there are
def printClassBreakdown(mascdb):
    print("Total flakes:\t\t\t" + str(len(mascdb.da)))
    print("Class Breakdown:")
    print("\tsmall_particle:\t\t\t" + str(len(mascdb.select_snowflake_class(1).da)))
    print("\tcolumnar_crystal:\t\t" + str(len(mascdb.select_snowflake_class(2).da)))
    print("\tplanar_crystal:\t\t\t" + str(len(mascdb.select_snowflake_class(3).da)))
    print("\taggregrate:\t\t\t" + str(len(mascdb.select_snowflake_class(4).da)))
    print("\tgraupel:\t\t\t" + str(len(mascdb.select_snowflake_class(5).da)))
    print("\tcolumnar_planar_combination:\t" + str(len(mascdb.select_snowflake_class(6).da)))

#Filter the images for image quality, class Prob, and size
def filterData(mascdb, quality=8.0, classProb=0.7):
    print("Filtering data to fit 300x300 image size")
    idx = mascdb.cam1['Dmax'] < 0.0075
    mascdb = mascdb.isel(idx)
    
    print("Filtering data by image quality, threshold: " + str(quality))
    idx = mascdb.triplet['flake_quality_xhi'] > quality
    mascdb = mascdb.isel(idx)
    
    print("Filtering data by class probability, threshold: " + str(classProb))
    idx = mascdb.triplet['snowflake_class_prob'] > classProb
    mascdb = mascdb.isel(idx)
    printClassBreakdown(mascdb)
    return mascdb

#Subset the data into class-balanced train and test sets
def subsetNoAug(mascdb, numPerClass, trainRatio):
    print("Subsetting data with no augmentation")
    numTrain = int((6*numPerClass) * trainRatio)
    numTest = (6*numPerClass) - numTrain

    class1 = mascdb.select_snowflake_class(1).head(numPerClass)
    class2 = mascdb.select_snowflake_class(2).head(numPerClass)
    class3 = mascdb.select_snowflake_class(3).head(numPerClass)
    class4 = mascdb.select_snowflake_class(4).head(numPerClass)
    class5 = mascdb.select_snowflake_class(5).head(numPerClass)
    class6 = mascdb.select_snowflake_class(6).head(numPerClass)
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        class1Images = class1.da[:, 362:662, 362:662, :]
        class2Images = class2.da[:, 362:662, 362:662, :]
        class3Images = class3.da[:, 362:662, 362:662, :]
        class4Images = class4.da[:, 362:662, 362:662, :]
        class5Images = class5.da[:, 362:662, 362:662, :]
        class6Images = class6.da[:, 362:662, 362:662, :]
    images = xarray.concat([class1Images, class2Images, class3Images, class4Images, class5Images, class6Images], dim="flake_id")
    images, labels = shuffle(images, getLabels(numPerClass), random_state=42)
    return images[:numTrain, :, :], labels[:numTrain], images[numTrain:, :, :], labels[numTrain:]

def getClass(mascdb, classNum, numPerClass):
    numClass = len(mascdb.select_snowflake_class(classNum).da)
    if(numClass < numPerClass):
        print("Augmenting class ", CLASSNAMES[classNum-1], " from ", numClass, " to ", numPerClass, "\n\tThis may take a while")
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            classSet = mascdb.select_snowflake_class(classNum).da[:, 362:662, 362:662, :]
        newImages = []
        tempImages = [classSet[n % numClass] for n in range(0, (numPerClass-numClass))]
        with ProcessPoolExecutor(max_workers=16) as executor:
            for i in executor.map(newImage, tempImages):
                newImages.append(i)
        newImages = xarray.concat(newImages, dim="flake_id")
        classImages = xarray.concat([newImages, classSet], dim="flake_id")
        print("Class augmentation complete")
        return classImages
    else:
        classSet = mascdb.select_snowflake_class(classNum).head(numPerClass)
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            return classSet.da[:, 362:662, 362:662, :]


#Perform the needed augmentation on a certain class
def aug(mascdb, classNum, numPerClass):
    numClass = len(mascdb.select_snowflake_class(classNum).da)
    if(numClass < numPerClass):
        print("Augmenting class " + CLASSNAMES[classNum-1] + " this will likely take a while")
        print("\tFeel free to bother Isaac telling him to make it faster, he's been neglecting doing so")
        classImages = mascdb.select_snowflake_class(classNum).da.stack(z=("flake_id", "cam_id")).transpose("z", ...)[:, 362:662, 362:662]
        newImages = []
        for i in range((3*numPerClass) - (3*numClass)):
            n = i % (3*numClass)
            newImages.append(newImage(classImages[n]))
        newImages = xarray.concat(newImages, dim="z")
        newImages = newImages.drop_vars("flake_id")
        newImages = newImages.drop_vars("cam_id")
        newImages = newImages.drop_vars("z")

        classImages = classImages.drop_vars("flake_id")
        classImages = xarray.concat([newImages, classImages], dim="z")
        print("Augmented class " + CLASSNAMES[classNum-1] + " from " + str(numClass) + " flakes to " + str(len(classImages)/3) + " flakes")
        return classImages
    else:
        print("No augmentation needed for class " + str(classNum))
        classSet = mascdb.select_snowflake_class(classNum).head(numPerClass)
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            temp = classSet.da.stack(z=("flake_id", "cam_id")).transpose("z", ...)[:, 362:662, 362:662]
            temp = temp.drop_vars("flake_id")
            return temp

#Subset the data into class-balanced train and test sets by augmenting classes with less data
def subsetAug(mascdb, numPerClass, trainRatio):
    print("Subsetting data with augmentation")
    numTrain = int((6*numPerClass) * trainRatio)
    numTest = (6*numPerClass) - numTrain

    class1Images = getClass(mascdb, 1, numPerClass)
    class2Images = getClass(mascdb, 2, numPerClass)
    class3Images = getClass(mascdb, 3, numPerClass)
    class4Images = getClass(mascdb, 4, numPerClass)
    class5Images = getClass(mascdb, 5, numPerClass)
    class6Images = getClass(mascdb, 6, numPerClass)

    images = xarray.concat([class1Images, class2Images, class3Images, class4Images, class5Images, class6Images], dim="flake_id")
    labels = getLabels(numPerClass)
    images, labels = shuffle(images, labels, random_state=42)
    return images[:numTrain, :, :], labels[:numTrain], images[numTrain:, :, :], labels[numTrain:]

#Split the data as needed into train and test set
def subsetData(mascdb, numPerClass, trainRatio):
    if not (min(numPerClass, len(mascdb.select_snowflake_class(6).da), len(mascdb.select_snowflake_class(3).da)) == numPerClass):
        print("WARNING, not enough flakes from each class, augmentation needed to balance")
        print("Requested: " + str(numPerClass) + " flakes from each class")
        printClassBreakdown(mascdb)
        return subsetAug(mascdb, numPerClass, trainRatio)
    else:
        return subsetNoAug(mascdb, numPerClass, trainRatio)
