import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("new_results.csv", dtype=str, delimiter=",")[1:,:]
shape_list = data[:,1]
shapes = np.unique(shape_list)
total_length = len(shape_list)
percentages = []
for s in shapes:
    amount = len(shape_list[np.where(shape_list==s)])
    percent = float(amount)/float(total_length) * 100.0
    print(s,":", percent)
    percentages.append(percent)

percentages = np.asarray(percentages)
fig1 = plt.figure(figsize=(20,20))
plt.pie(percentages,labels=shapes,autopct='%1.2f%%',textprops={'fontsize': 20})
plt.show()
