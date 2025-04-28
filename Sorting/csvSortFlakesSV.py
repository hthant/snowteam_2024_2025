#import sklearn.metrics import confusion_matrix
import csv
import numpy as np
import time
import os
from singleView import loadModel
from updatedUtils import getPrediction, getInput

# Paths to the weights for shape, riming, and melting classifications
shape_weights = '/home/nvidia/SnowTeam/SnowTeam_AGX/weights/singleview/shape-0.95.hdf5'
riming_weights = '/home/nvidia/SnowTeam/SnowTeam_AGX/weights/singleview/riming-0.95.hdf5'
melting_weights = '/home/nvidia/SnowTeam/SnowTeam_AGX/weights/singleview/melting-0.83.hdf5'

# Path to the image folder (change depending on folder name)
path_to_image_folder = '/home/nvidia/SnowTeam/SnowTeam_AGX/newSMASImages/'
imageSize = 300
csv_output_file = '/home/nvidia/SnowTeam/SnowTeam_AGX/classificationResults/newSMASResults.csv'
skipped_images_file = '/home/nvidia/SnowTeam/SnowTeam_AGX/classificationResults/skippedImages.csv'

start_time = time.time()

# Load models for each classification type
shape_model = loadModel(6, shape_weights, configFile='', imageSize=imageSize, dropout=0.3, learningRate=0.0001)
riming_model = loadModel(5, riming_weights, configFile='', imageSize=imageSize, dropout=0.3, learningRate=0.0001)
melting_model = loadModel(2, melting_weights, configFile='', imageSize=imageSize, dropout=0.3, learningRate=0.0001)

images = os.listdir(path_to_image_folder)
print("found: " + str(len(images)) + " images in: " + path_to_image_folder)

# Labels for each classification type
shape_labels = ['small_particles', 'columnar_crystals', 'planar_crystals', 'aggregrates', 'graupel', 'columnar_planar_combination']
riming_labels = ['unrimed', 'rimed', 'densely_rimed', 'graupel_like', 'graupel']
melting_labels = ['dry', 'wet']

# Open the CSV file for writing
with open(csv_output_file, mode='w', newline='') as csv_file, open(skipped_images_file, mode='w', newline='') as skipped_file:
    writer = csv.writer(csv_file)
    skipped_writer = csv.writer(skipped_file)
    # Write the header
    writer.writerow(['Image Name', 'Shape', 'Riming', 'Melting'])
    skipped_writer.writerow(['Image Name', 'Reason'])

    print("Starting classification")
    for flake in images:
        if ".png" in flake:
            # Preprocess image
            img = getInput(path_to_image_folder + flake, imageSize=imageSize)
            
            # Get predictions for shape, riming, and melting
            shape_pred = getPrediction(img, shape_model, threshold=0.8)
            riming_pred = getPrediction(img, riming_model, threshold=0.8)
            melting_pred = getPrediction(img, melting_model, threshold=0.8)
            
            # Only save predictions if all classifications meet the threshold
            if shape_pred is None or riming_pred is None or melting_pred is None:
                # Log skipped image and reason(s)
                reasons = []
                if shape_pred is None:
                    reasons.append("shape below threshold")
                if riming_pred is None:
                    reasons.append("riming below threshold")
                if melting_pred is None:
                    reasons.append("melting below threshold")
                skipped_writer.writerow([flake, "; ".join(reasons)])
                print(f"Skipping {flake} - {', '.join(reasons)}")
            else:
                # Get predicted labels
                shape_class = shape_labels[shape_pred]
                riming_class = riming_labels[riming_pred]
                melting_class = melting_labels[melting_pred]
            
                # Write the image name and predicted classes to the CSV
                writer.writerow([flake, shape_class, riming_class, melting_class])

print(f"----execution took {time.time()-start_time} seconds----")
print(f"Classification results saved to {csv_output_file}")
print(f"Skipped images logged in {skipped_images_file}")