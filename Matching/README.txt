===================================================================================================
Matching
===================================================================================================

This code reads in files from up to 7 cameras after processing from the cropping / S3 code and then matches
them into pairs, from there it attempts to combine pairs into groups with as many angles of the same
snowflake as possible. Before completing the matching fundamental matrix have to be calculated which 
calibrates the cameras for the matching code.

===================================================================================================
Files
===================================================================================================

Config.json: The config file, allows you to change all of the file paths of inputs and outputs, enable / 
             disable all camera pairs and change the output of the program. It also allows the modification
             of regex expressions if those change in the future

fundamentalMatrix.json: Stores the fundamental matrix values of all of the camera pairs if you dont want to
                        calculate them every time

src/main.cc: This code runs the whole program, it reads in the config files, runs fundamental matrix
             calculations, performs matching, and then combines the outputs into one place

fundamentalMatrix.cc/.h: Reads in all of the files associated with the calibration images and produces the 
                         fundamental matrix for that camera pairs
                        
matchingCode.cc/.h: Looks through all of the images between two cameras and based on time stamps, snow flake
                    positioning, and fundamental matrix and determines if there are matches. These matches
                    are written to a file unique to the camera pairs

utils.cc/.h: This file contains utility functions that are used in the other files.

===================================================================================================
Usage
===================================================================================================

To compile:             (/matching/src)
    make run

To clean:               (/matching/src)
    make clean

To run:                 (/matching/bin)
    ./snowMatching


