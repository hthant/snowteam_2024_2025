import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers

def readFile(path):
    f = open(path, 'r')
    data = f.read()
    return data.split('], [')

def findNth(s, ss, n):
    sep = s.split(ss, n)
    if len(sep) <= n:
        return -1
    return len(s) - len(sep[-1]) - len(ss)

def cleanList(l):
    out = []
    for i in range(0, len(l)):
        l1 = l[i][l[i].index('/home'):l[i].index('.png')+4]
        l2 = l[i][findNth(l[i], '/home', 2):findNth(l[i], '.png', 2)+4]

        out.append((l1, l2))
    return out

def compileList(list1, list2):
    out = []
    for l1 in enumerate(list1):
        for l2 in enumerate(list2):
            if l1[0] == l2[0]:
                out.append((l1[0], l1[1], l2[1]))
    return out

def getList(path1, path2, v=0):
    list1 = cleanList(readFile(path1))
    list2 = cleanList(readFile(path2))
    list3 = compileList(list1, list2)

    if v == 1:
        print("Number of pairs found between first pair of cameras: ", len(list1))
        print("Number of pairs found between second pair of cameras: ", len(list2))
        print("Number of triplets fround across all three cameras: ", len(list3))
        print(list3)
    return list3

def getInput(flake, imageSize=300):
    img = np.asarray(Image.open(flake))
    img = tf.cast(img, dtype=tf.float32)
    img = tf.math.subtract(img, tf.math.reduce_mean(img))
    img = tf.math.divide(img, tf.math.reduce_std(img))
    img = tf.stack([img, img, img])
    img = tf.reshape(img, [1, imageSize, imageSize, 3])
    return img

def getInput3Channels(flakes, imageSize=300):
    return getInput(flakes[0], imageSize=imageSize), getInput(flakes[1], imageSize=imageSize), getInput(flakes[2], imageSize=imageSize)

# Return prediction of the model on the provided image
def getPrediction(image, model):
  return np.argmax(model.predict(image, verbose=0))

# Return prediction of the model on the imagePath specified
def getPredictionFromPath(imagePath, model):
  return np.argmax(model.predict(np.asarray(Image.open(imagePath)), verbose=0))

# Return needed layer type
def getLayer(line):
    if "ropout" in line or "do" in line or "DO" in line:
        return layers.Dropout(float(line[line.index("(")+1:line.index(")")]))
    elif "ense" in line or "FC" in line or "fc" in line:
        return layers.Dense(int(line[line.index("(")+1:line.index(")")]), use_bias=False)
    elif "BN" in line or "normalization" in line or "bn" in line:
        return layers.BatchNormalization(center=True, scale=False)
    else: print("Error check classification head config, layer type not found: ", line)
