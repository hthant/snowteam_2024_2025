# Sorting

Once the images have been cropped and in the case of mutliview sorted we can use pretrained models to classify them and sort the cropped images. The models are trained [here](https://github.com/Isaac-Jacobson/snowClassification/tree/main).

Sorting cropped flakes using a single view classifier with sortFlakes_singleView.py needs the path to the images, the path to the desired weights folder, the number of classes of the classifier, and the image size. sortFlakes_multiView.py also requires the paths to the 2 output text files of sortingCode.py. Also before running the code, make sure that the corrrect list of outputDirs is uncommented, this may need to be edited in the case of 5 class shape classification or 4 class riming classification. 
