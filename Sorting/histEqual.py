import cv2
import os

# Define the paths
input_folder = '/home/nvidia/SnowTeam/SnowTeam_AGX/smasImages'
output_folder = '/home/nvidia/SnowTeam/SnowTeam_AGX/histEqSMAS'
os.makedirs(output_folder, exist_ok=True)

# Loop over each image in the input folder
for image_name in os.listdir(input_folder):
    if image_name.endswith('.png'):
        # Read the image in grayscale (you can adjust this if using color images)
        img_path = os.path.join(input_folder, image_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply histogram equalization
        equalized_img = cv2.equalizeHist(img)
        
        # Save the processed image to the output folder
        output_path = os.path.join(output_folder, image_name)
        cv2.imwrite(output_path, equalized_img)
        print(f"Processed and saved {image_name}")

print("Histogram equalization completed for all images.")