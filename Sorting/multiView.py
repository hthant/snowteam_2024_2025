import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications.inception_v3 import InceptionV3
from utils import getLayer

# ==============================================================================
# Models
# ==============================================================================

# Multi view inception based feauture extractor
def getFeatureExtractor(imageSize=300, imagenet=True):
    input1 = layers.Input(shape=(300, 300, 3))
    input2 = layers.Input(shape=(300, 300, 3))
    input3 = layers.Input(shape=(300, 300, 3))

    if imageSize == 300 and imagenet:
        print("feature extractor pretrained on imagenet dataset")
        base_model1 = InceptionV3(include_top = False, weights="imagenet")
        base_model2 = InceptionV3(include_top = False, weights="imagenet")
        base_model3 = InceptionV3(include_top = False, weights="imagenet")
    else:
        base_model1 = InceptionV3(include_top = False)
        base_model2 = InceptionV3(include_top = False)
        base_model3 = InceptionV3(include_top = False)


    base_model1._name = "featureExtractor1"
    base_model1 = base_model1(input1)
    base_model1 = layers.GlobalAveragePooling2D()(base_model1)
    
    base_model2._name = "featureExtractor2"
    base_model2 = base_model2(input2)
    base_model2 = layers.GlobalAveragePooling2D()(base_model2)
    
    base_model3._name = "featureExtractor3"
    base_model3 = base_model3(input3)
    base_model3 = layers.GlobalAveragePooling2D()(base_model3)
    
    base_model = layers.concatenate([base_model1, base_model2, base_model3], axis=1)
    base_model = layers.Flatten()(base_model)
    
    return base_model, input1, input2, input3
    
# Make a custom classification head
def getClassificationHead(numClasses, featureExtractor, configFile='', dropout=0.3):
    if configFile == '':
        print("Classification head configuration not specified, using deafault")
        model = layers.Dense(256, use_bias=False)(featureExtractor)
        model = layers.BatchNormalization(center=True, scale=False)(model)
        model = layers.Dropout(dropout)(model)

        model = layers.Dense(128, use_bias=False)(model)
        model = layers.BatchNormalization(center=True, scale=False)(model)
        model = layers.Dropout(dropout)(model)

        model = layers.Dense(64, use_bias=False)(model)
        model = layers.BatchNormalization(center=True, scale=False)(model)
        model = layers.Dropout(dropout)(model)

        model = layers.Dense(numClasses, activation="softmax")(model)
        return model
    else:
        f = open(configFile, 'r')
        lines = f.readlines()
        model = getLayer(lines[0])(featureExtractor)
        for line in lines[1:]:
            model = getLayer(line)(model)
        model = layers.Dense(numClasses, activation="softmax")(model)
        return model


# Make a complete model
def makeModel(numClasses, configFile='', imageSize=300, dropout=0.3):
    fe, i1, i2, i3 = getFeatureExtractor(imageSize)
    return Model(inputs=[i1, i2, i3], outputs=getClassificationHead(numClasses, fe, configFile=configFile, dropout=dropout))

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
        print("Multi view inception model loaded with exsisting weights\nclassification_head:\t[dense( 256 ) -> dr(",dropout,") -> dense( 128 ) -> dr(",dropout,") -> dense( 32 ) -> dr(",dropout,") -> softmax(",numClasses,")]\nlearning_rate:\t", learningRate, "\nimage_size:\t", imageSize,"\nweights:\t", weightsPath)
    else:
        print("Multi view inception model loaded with exsisting weights\nclassification_head loaded from, ", configFile, "\nweights:\t", weightsPath)
    return model
