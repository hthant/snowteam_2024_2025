import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

prefix = "../../combImagesEq/"
old_csv = "new_results.csv" #"new_results.csv"
new_csv = "new_results.csv"
#question = 'aggregrates'
question = 'columnar_planar_combination'
#question = 'small_particles'

cutoff = 200
shuffleBool = True

data = np.loadtxt(old_csv,delimiter=",",dtype=str)
print(data.shape)
example = data[data[:,1]==question]
remainder = data[data[:,1]!=question]
print(remainder.shape)
if shuffleBool:
    np.random.shuffle(example)
updated_example = []
example_untouched = example[cutoff:]
for i,ex in enumerate(example[:cutoff]):
    print(i,prefix+ex[0])
    fig = plt.figure()
    plt.imshow(Image.open(prefix+ex[0]), cmap='gray')
    plt.show()
   #plt.waitforbuttonpress(0)
    plt.close('all')
    
    while True:
        inputOptions = str(input(ex[0]+"| good:0, SP:1, CC:2, PC:3, AG:4, GR:5, CPC:6 - "))
        print(inputOptions)
        if(inputOptions == "0" or inputOptions == "1" or inputOptions == "2" or inputOptions == "3" or inputOptions == "4" or inputOptions == "5" or inputOptions == "6"):
            break
        else:
            print("wrong input, pick 0-6")
    inputOptions = int(inputOptions)
    if inputOptions == 1:
        ex[1] = "small_particles"
    elif inputOptions == 2:
        ex[1] = "columnar_crystals"
    elif inputOptions == 3:
        ex[1] = "planar_crystals"
    elif inputOptions == 4:
        ex[1] = "aggregrates"
    elif inputOptions == 5:
        ex[1] = "graupel"
    elif inputOptions == 6:
        ex[1] = "columnar_planar_combination"
    updated_example.append(ex)

updated_example = np.asarray(updated_example)
allData = np.vstack((remainder, np.vstack((updated_example, example_untouched))))
print(allData.shape)

np.savetxt(new_csv, allData, delimiter=",",fmt='%s')
