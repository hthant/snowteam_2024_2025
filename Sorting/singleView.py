import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications.inception_v3 import InceptionV3
from utils import getLayer

# ==============================================================================
# Models
# ==============================================================================

# Single view inception based feauture extractor
def getFeatureExtractor(imageSize=300, imagenet=True):
    if imageSize == 300 and imagenet:
        print("feature extractor pretrained on imagenet dataset")
        base_model = InceptionV3(input_shape = (imageSize, imageSize, 3), include_top = False, weights="imagenet")
    else: 
        base_model = InceptionV3(input_shape = (imageSize, imageSize, 3), include_top = False)
    add_model = Sequential()
    add_model.add(base_model)
    add_model.add(layers.GlobalAveragePooling2D())
    return add_model

# Make a custom classification head
def getClassificationHead(numClasses, featureExtractor, configFile='', dropout=0.3):
    if configFile=='':
        print("Classification head configuration not specified, using deafault")
        featureExtractor.add(layers.Dense(256, use_bias=False))
        featureExtractor.add(layers.BatchNormalization(center=True, scale=False))
        featureExtractor.add(layers.Dropout(dropout))

        featureExtractor.add(layers.Dense(128, use_bias=False))
        featureExtractor.add(layers.BatchNormalization(center=True, scale=False))
        featureExtractor.add(layers.Dropout(dropout))

        featureExtractor.add(layers.Dense(64, use_bias=False))
        featureExtractor.add(layers.BatchNormalization(center=True, scale=False))
        featureExtractor.add(layers.Dropout(dropout))

        featureExtractor.add(layers.Dense(numClasses, activation="softmax"))
        return featureExtractor
    else:
        f = open(configFile, 'r')
        for line in f.readlines():
            featureExtractor.add(getLayer(line))
        featureExtractor.add(layers.Dense(numClasses, activation="softmax"))
        return featureExtractor

# Make a complete model
def makeModel(numClasses, configFile='', imageSize=300, dropout=0.3) :
    fe = getFeatureExtractor(imageSize)
    return getClassificationHead(numClasses, fe, configFile=configFile, dropout=dropout)

# Return a compiled model loaded with the provided weights
def loadModel(numClasses, weightsPath, configFile='', imageSize=300, dropout=0.3, learningRate=0.0001):
    devices_names = [d.name.split("e:")[1] for d in tf.config.list_physical_devices('GPU')]
    mirrored_strategy = tf.distribute.MirroredStrategy(devices=devices_names[:1]) 
    with mirrored_strategy.scope():
        model = makeModel(numClasses, configFile=configFile, imageSize=imageSize, dropout=dropout)
        model.load_weights(weightsPath)
        model.compile(optimizer = RMSprop(learning_rate=learningRate), 
                        loss = 'binary_crossentropy', 
                        metrics = ['acc'])
    if configFile=='':
        print("Single view inception model loaded with exsisting weights\nclassification_head:\t[dense( 256 ) -> dr(",dropout,") -> dense( 128 ) -> dr(",dropout,") -> dense( 32 ) -> dr(",dropout,") -> softmax(",numClasses,")]\nlearning_rate:\t", learningRate, "\nimage_size:\t", imageSize,"\nweights:\t", weightsPath)
    else:
        print("Single view inception model loaded with exsisting weights\nclassification_head loaded from, ", configFile, "\nweights:\t", weightsPath)
    return model
