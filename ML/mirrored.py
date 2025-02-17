import xarray
import xbatcher

import pickle
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.optimizers import RMSprop
#from tensorflow.keras.applications.efficientnet import EfficientNetB3
#from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint
from os.path import exists
from tensorflow.keras.callbacks import Callback
import os
from discord_webhook import DiscordWebhook as dw
from datetime import datetime as dt

from utils3 import *
from toKeras3 import *

BATCHSIZE = 180
NUMPERCLASS = 480
TRAINSPLIT = 0.8
EPOCHS = 7

url = "https://discord.com/api/webhooks/1306737450793570355/85IQUfMzkDEDJMSPWKJh1UfBcXFWWncCf5VOksut5KYnzgtZFF-12w8_Dniix66Fj0yJ"

'''
TO-DO:
Create testbench environment for Mirrored and Central Storage strategies
    - Increase batch size to large number, and use as little NUMPERCLASS as possible
    - Include method of storing GPU performance into csv files
Organize code files and begin documentation procedure
'''

def getENetB3():
    input_shape1 = layers.Input(shape=(300, 300, 3))
    input_shape2 = layers.Input(shape=(300, 300, 3))
    input_shape3 = layers.Input(shape=(300, 300, 3))
    
    ceppy = InceptionV3(include_top = False, weights="imagenet")
    #for l in ceppy.layers:
    #    l.trainable = False
    ceppy._name = "featureExtractor1"
    base_model1 = ceppy(input_shape1)
    base_model1 = layers.GlobalAveragePooling2D()(base_model1)
    
    ceppy2 = InceptionV3(include_top = False, weights="imagenet")
    #for l in ceppy2.layers:
    #    l.trainable = False
    ceppy2._name = "featureExtractor2"
    base_model2 = ceppy2(input_shape2)
    base_model2 = layers.GlobalAveragePooling2D()(base_model2)
    
    base_model3 = InceptionV3(include_top = False, weights="imagenet")
    base_model3._name = "featureExtractor3"
    #for l in base_model3.layers:
    #    l.trainable = False
    base_model3 = base_model3(input_shape3)
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

def getModel(d_num=1):
    devices = tf.config.list_physical_devices('GPU')

    devices_names = [d.name.split("e:")[1] for d in devices]
    if (d_num == 3):
        mirrored_strategy = tf.distribute.MirroredStrategy(devices=devices_names)
    else:
        mirrored_strategy = tf.distribute.MirroredStrategy(devices=devices_names[:d_num])
    with mirrored_strategy.scope():
        model = getENetB3()
        model.compile(optimizer = RMSprop(learning_rate=0.0001), 
                        loss = 'binary_crossentropy', 
                        metrics = ['acc'])
    return model
'''
 # Make only 1st GPU visible to Tensorflow
    tf.config.set_visible_devices(devices[0], 'GPU')
    
    # Set memory growth to avoid allocation issues
    tf.config.experimental.set_memory_growth(devices[0], True)
'''
'''
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

        # Creating filepath with metrics
        if val_acc is not None:
            current_file_path = self.filepath.format(
                epoch=epoch + 1,
                val_acc=val_acc,
                train_acc=train_acc,
                val_loss=val_loss,
                train_loss=train_loss
            )
            
            # Save only if validation accuracy improves
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.model.save(current_file_path)
                print(f"\nModel improved and saved to {current_file_path}")
'''
def train(model, train_generator, validation_generator):
    filepath='/home/snow/nashein/hthant/weights/effnetb3/shape/w-{epoch:02d}-va-{val_acc:.6f}-vl-{val_loss:.6f}-ta-{train_acc:.6f}-tl-{train_loss:.6f}.tf'
    #callbacks = CustomModelCheckpoint(filepath=filepath)
    history = model.fit(
            train_generator,
            validation_data = validation_generator,
            epochs = EPOCHS,
            steps_per_epoch= int(((NUMPERCLASS*6)*TRAINSPLIT)/BATCHSIZE),
            validation_steps = int(((NUMPERCLASS*6)*(1-TRAINSPLIT))/BATCHSIZE),
            verbose = 1
            #callbacks=[callbacks]
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
    for i in range(2, 4):
        model = getModel(d_num=i)
        time_before = dt.now()
        model, history = train(model, train_generator, test_generator)
        time_after = dt.now()
        time_dif = time_after - time_before
        minutes = divmod(time_dif.total_seconds(), 60)
        print("Training complete! Total minutes taken to train {} epochs on {} GPUs with {} images: {:.2f} minutes".format(EPOCHS, 1, NUMPERCLASS, minutes[0]))
    webhook = dw(url=url, content="ML Training Done")
    # 3 GPUs: 227 minutes
    # 2 GPUs: 219 minutes
    # 1 GPU : 219 minutes
    response = webhook.execute()

# Any errors are caught here
except Exception as e:
    print(e)
    webhook = dw(url=url, content="ML Training Errored Out")
    response = webhook.execute()

print("\nfin\n")
