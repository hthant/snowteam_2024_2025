import xarray
import xbatcher

import pickle
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.optimizers import RMSprop
#from tensorflow.keras.applications.efficientnet import EfficientNetB3
#from tensorflow.keras.applications.efficientnet import preprocess_input
#from tensorflow.keras.applications.inception_v3 import InceptionV3
#from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint
from os.path import exists
from tensorflow.keras.callbacks import Callback
import os
from discord_webhook import DiscordWebhook as dw
from datetime import datetime as dt
import csv

from utils3 import *
from toKeras3 import *

#BATCHSIZE = 204
NUMPERCLASS = 480
TRAINSPLIT = 0.8
EPOCHS = 8

url = "https://discord.com/api/webhooks/1306737450793570355/85IQUfMzkDEDJMSPWKJh1UfBcXFWWncCf5VOksut5KYnzgtZFF-12w8_Dniix66Fj0yJ"

'''
TO-DO:
Create testbench environment for Mirrored and Central Storage strategies
    - Increase batch size to large number, and use as little NUMPERCLASS as possible
    - Include method of storing GPU performance into csv files
Organize code files and begin documentation procedure
Find a way to test code modularity regarding pre-trained models, then compare their metrics to InceptionV3 model
'''

def getML(ML_type=0):
    input_shape1 = layers.Input(shape=(300, 300, 3))
    input_shape2 = layers.Input(shape=(300, 300, 3))
    input_shape3 = layers.Input(shape=(300, 300, 3))
    
    # Initial model with InceptionV3 loaded with imagenet (default) weights
    if (ML_type == 0):
        from tensorflow.keras.applications.inception_v3 import InceptionV3
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        ceppy = InceptionV3(include_top = False, weights="imagenet")
        ceppy2 = InceptionV3(include_top = False, weights="imagenet")
        ceppy3 = InceptionV3(include_top = False, weights="imagenet")
    
    # EfficientNetB6
    elif (ML_type == 1):
        from tensorflow.keras.applications.efficientnet import EfficientNetB6
        from tensorflow.keras.applications.efficientnet import preprocess_input
        ceppy = EfficientNetB6(include_top = False, weights="imagenet")
        for layer in ceppy.layers[:450]:
            layer.trainable = False
        ceppy2 = EfficientNetB6(include_top = False, weights="imagenet")
        for layer in ceppy2.layers[:450]:
            layer.trainable = False
        ceppy3 = EfficientNetB6(include_top = False, weights="imagenet")
        for layer in ceppy3.layers[:450]:
            layer.trainable = False
    
    # EfficientNetV2B3
    elif (ML_type == 2):
        from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B3
        from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
        ceppy = EfficientNetV2B3(include_top = False, weights="imagenet")
        for layer in ceppy.layers[:450]:
            layer.trainable = False
        ceppy2 = EfficientNetV2B3(include_top = False, weights="imagenet")
        for layer in ceppy2.layers[:450]:
            layer.trainable = False
        ceppy3 = EfficientNetV2B3(include_top = False, weights="imagenet")
        for layer in ceppy3.layers[:450]:
            layer.trainable = False
    
    # ConvNeXtBase
    elif (ML_type == 3):
        from tensorflow.keras.applications.convnext import ConvNeXtBase
        from tensorflow.keras.applications.convnext import preprocess_input
        ceppy = ConvNeXtBase(include_top = False, weights="imagenet")
        for layer in ceppy.layers[:450]:
            layer.trainable = False
        ceppy2 = ConvNeXtBase(include_top = False, weights="imagenet")
        for layer in ceppy2.layers[:450]:
            layer.trainable = False
        ceppy3 = ConvNeXtBase(include_top = False, weights="imagenet")
        for layer in ceppy3.layers[:450]:
            layer.trainable = False
    
    # Unrecognized ML_type value
    else:
        print("No matching ML model found for your variable, or the ML_type is set to an unknown value.")
        print("You can add more ML models by editing the code. The Keras Applications (link below) should prove helpful.")
        print("https://keras.io/api/applications/")
    
    #ceppy = EfficientNetB3(include_top = False, weights="imagenet")
    #for l in ceppy.layers:
    #    l.trainable = False
    ceppy._name = "featureExtractor1"
    base_model1 = ceppy(input_shape1)
    base_model1 = layers.GlobalAveragePooling2D()(base_model1)
    
    #ceppy2 = EfficientNetB3(include_top = False, weights="imagenet")
    #for l in ceppy2.layers:
    #    l.trainable = False
    ceppy2._name = "featureExtractor2"
    base_model2 = ceppy2(input_shape2)
    base_model2 = layers.GlobalAveragePooling2D()(base_model2)
    
    #base_model3 = EfficientNetB3(include_top = False, weights="imagenet")
    #for l in base_model3.layers:
    #    l.trainable = False
    ceppy3._name = "featureExtractor3"
    base_model3 = ceppy3(input_shape3)
    base_model3 = layers.GlobalAveragePooling2D()(base_model3)
    
    base_model = layers.concatenate([base_model1, base_model2, base_model3], axis=1)
    base_model = layers.Flatten()(base_model)

    #base_model.trainable = False

    #add_model = (layers.GlobalAveragePooling2D())(base_model)
    #add_model = Sequential()(base_model)
    add_model = (layers.Dense(512, use_bias=False))(base_model)
    add_model = (layers.BatchNormalization(center=True, scale=False))(add_model)
    add_model = (layers.Dropout(0.35))(add_model)


    add_model = (layers.Dense(64, use_bias=False))(add_model)
    add_model = (layers.BatchNormalization(center=True, scale=False))(add_model)
    add_model = (layers.Dropout(0.3))(add_model)


    add_model = (layers.Dense(16, use_bias=False))(add_model)
    add_model = (layers.BatchNormalization(center=True, scale=False))(add_model)
    add_model = (layers.Dropout(0.25))(add_model)

    add_model = (layers.Dense(6, activation="softmax"))(add_model)

    add_model = Model(inputs=[input_shape1, input_shape2, input_shape3], outputs=add_model)
    return add_model

def getModel():
    devices = tf.config.list_physical_devices('GPU')
    devices_names = [d.name.split("e:")[1] for d in devices]
    
    # Previous code for selecting number of GPUs available, kept for reference
    '''
    if (d_num == 3):
        mirrored_strategy = tf.distribute.MirroredStrategy(devices=devices_names)
    else:
        mirrored_strategy = tf.distribute.MirroredStrategy(devices=devices_names[:d_num])
    '''

    mirrored_strategy = tf.distribute.MirroredStrategy(devices=devices_names)
    with mirrored_strategy.scope():
        model = getML(ML_type=1)
        model.compile(optimizer = RMSprop(learning_rate=0.0001), 
                        loss = 'binary_crossentropy', 
                        metrics = ['acc'])
    return model

class CSVLogger(Callback):
    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = csv_path

        # Write the header row
        with open(self.csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_acc = logs.get("acc", logs.get("accuracy"))
        train_loss = logs.get("loss")
        val_acc = logs.get("val_acc", logs.get("val_accuracy"))
        val_loss = logs.get("val_loss")

        # Append row with metrics for the current epoch
        with open(self.csv_path, mode='a+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, train_loss, train_acc, val_loss, val_acc])

class CustomModelCheckpoint(Callback):
    def __init__(self, filepath, **kwargs):
        super().__init__()
        self.filepath = filepath
        self.best_val_acc = -float("inf")
        self.kwargs = kwargs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Accessing training and validation metrics
        train_acc = logs.get("acc", logs.get("accuracy"))
        train_loss = logs.get("loss")
        val_acc = logs.get("val_acc", logs.get("val_accuracy"))
        val_loss = logs.get("val_loss")

        # Creating unique filepath with metrics
        current_file_path = self.filepath.format(
            epoch=epoch + 1,
            val_acc=val_acc,
            train_acc=train_acc,
            val_loss=val_loss,
            train_loss=train_loss,
            timestamp=dt.now().strftime("%Y/%m/%d-%H:%M:%S")  # add timestamp for uniqueness
        )
        
        # Save only if validation accuracy improves
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            
            # Try to delete existing file, handle potential errors
            try:
                if os.path.exists(current_file_path):
                    os.remove(current_file_path)
                self.model.save(current_file_path, overwrite=True)
                print(f"\nModel improved and saved to {current_file_path}")
            except Exception as e:
                print(f"\nError during saving: {e}")

def train(model, train_generator, validation_generator):
    filepath = '/home/snow/nashein/hthant/weights/effnetb3/shape/jasem-w-{epoch:02d}.hdf5'
    csv_path = "/home/snow/smij/csv/j_training_metrics.csv"
    callbacks = [CustomModelCheckpoint(filepath=filepath), CSVLogger(csv_path=csv_path)]
    history = model.fit(
            train_generator,
            validation_data = validation_generator,
            epochs = EPOCHS,
            steps_per_epoch= int(((NUMPERCLASS*6)*TRAINSPLIT)/BATCHSIZE),
            validation_steps = int(((NUMPERCLASS*6)*(1-TRAINSPLIT))/BATCHSIZE),
            verbose = 1,
            callbacks=[callbacks]
        )
    return model, history

path1 = ''
path2 = ''

# Midway point between 1000 and 2000, rounds down to save time by excluding image augmentation
if(NUMPERCLASS < 1500):
    NUMPERCLASS = 1000
    path1 = '/home/snow/nashein/hthant/weights/effnetb3/imageCheckPoint/1000_training_shape.pkl'
    path2 = '/home/snow/nashein/hthant/weights/effnetb3/imageCheckPoint/1000_test_shape.pkl'

# Midway point between 2000 and 3000, rounds down to save time by excluding image augmentation
elif(NUMPERCLASS < 2500):
    NUMPERCLASS = 2000
    path1 = '/home/snow/nashein/hthant/weights/effnetb3/imageCheckPoint/2000_training_shape.pkl'
    path2 = '/home/snow/nashein/hthant/weights/effnetb3/imageCheckPoint/2000_test_shape.pkl'

# Midway point between 3000 and 4800, rounds down to save time by excluding image augmentation
elif(NUMPERCLASS < 3900):
    NUMPERCLASS = 3000
    path1 = '/home/snow/nashein/hthant/weights/effnetb3/imageCheckPoint/3000_training_shape.pkl'
    path2 = '/home/snow/nashein/hthant/weights/effnetb3/imageCheckPoint/3000_test_shape.pkl'

# Any NUMPERCLASS that is 3900 or higher, use initial pickle files with 4800 augmented images
else:
    NUMPERCLASS = 4800
    path1 = '/home/snow/nashein/hthant/weights/effnetb3/imageCheckPoint/training_shape.pkl'
    path2 = '/home/snow/nashein/hthant/weights/effnetb3/imageCheckPoint/test_shape.pkl'

if(exists(path1) and exists(path2)):
    file1 = open(path1, 'rb')
    training = pickle.load(file1)
    train_images, train_labels = training
    file1.close()

    file2 = open(path2, 'rb')
    testing = pickle.load(file2)
    test_images, test_labels = testing
    file2.close()
else:
    train_images, train_labels, test_images, test_labels = subsetData(filterData(loadData("/home/snow/masc_data/")), NUMPERCLASS, TRAINSPLIT)
    training = train_images, train_labels
    testing = test_images, test_labels

    file1 = open(path1, 'wb')
    pickle.dump(training, file1)
    file1.close()

    file2 = open(path2, 'wb')
    pickle.dump(testing, file2)
    file2.close()

print("Generating training and testing images...")
train_images_generator = xbatcher.BatchGenerator(train_images,input_dims={'flake_id':BATCHSIZE})
test_images_generator = xbatcher.BatchGenerator(test_images,input_dims={'flake_id':BATCHSIZE})
train_labels_generator = xbatcher.BatchGenerator(train_labels,input_dims={'l':BATCHSIZE})
test_labels_generator = xbatcher.BatchGenerator(test_labels,input_dims={'l':BATCHSIZE})

train_generator = CustomTFDataset(train_images_generator, train_labels_generator, transform=transformTensor, target_transform=transformLabel)
test_generator = CustomTFDataset(test_images_generator, test_labels_generator, transform=transformTensor, target_transform=transformLabel)
print("Generation complete, now training model...")

# Check if training proccess produces any errors, sending a message to Discord
try:
    gput = []
    for i in range(0, 10):
        model = getModel()
        time_before = dt.now()
        model, history = train(model, train_generator, test_generator)
        time_after = dt.now()
        time_dif = time_after - time_before
        minutes = divmod(time_dif.total_seconds(), 60)
        #print("Training complete! Total minutes taken to train {} epochs on {} GPUs with {} images: {:.2f} minutes".format(EPOCHS, 1, NUMPERCLASS, minutes[0]))
        gput.append(minutes)
    webhook = dw(url=url, content="ML Training Done")
    # 3 GPUs: 227 minutes
    # 2 GPUs: 219 minutes
    # 1 GPU : 219 minutes
    response = webhook.execute()
    print(gput)

# Any errors are caught here
except Exception as e:
    print(e)
    webhook = dw(url=url, content="ML Training Errored Out")
    response = webhook.execute()

print("\nfin\n")
