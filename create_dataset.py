import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math

# accpath = 'Accelerometer_Train/'
# gypath = 'Gyroscope_Train/'
# magpath = 'Magnetometer_Train/'

save_folder = 'data/TRAIN/'


def getListOfFiles(dirName, LA, LG, LM, label_a, label_g, label_m):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    i = 0
    for entry in listOfFile:
        # Create full path
        ch = dirName[-2] + dirName[-1]
        fullPath = os.path.join(dirName, entry)
        
        if os.path.isdir(fullPath):
            af, LA, LG, LM, label_a, label_g, label_m = getListOfFiles(fullPath, LA, LG, LM, label_a, label_g, label_m)
            allFiles = allFiles + af

   #      elif entry == 'BMI160_ACCELEROMETER_Accelerometer_Wakeup.csv':
			# print(fullPath)
			# r = convert_to_npy(fullPath,0,0)
			# LA.append(r)
			# l = dirName[6:8]
			# label_a.append((int)(l))
			# print(accpath + dirName[6:8] + '/' + ch +  '.npy')
        	
   #      elif entry == 'BMI160_GYROSCOPE_Gyroscope_Wakeup.csv':
			# print(fullPath)
			# r = convert_to_npy(fullPath,1,0)
			# LG.append(r)
			# l = dirName[6:8]
			# label_g.append((int)(l))
			# print(gypath + dirName[6:8] + '/' + ch +  '.npy')

        elif entry == 'BMI160_ACCELEROMETER_Accelerometer_Wakeup.npy':
			print(fullPath)
			# r = convert_to_npy(fullPath,2,0)
			arr = np.load(fullPath)
			arr = np.transpose(arr)
			l = dirName[10:12]
			label_m.append((int)(l))
			# LM.append(r)
			# print(magpath + dirName[6:8] + '/' + '.npy')

			si, ei = 0, 150 # 30
			li = arr.shape[0]
			i = 1
			while ei < li:
				ln = []
				for x in range(si, ei):
					[a,b,c] = arr[x]
					s = a**2 + b**2 + c**2
					sq = math.sqrt(s)
					ln.append([a/sq, b/sq, c/sq, sq])
				si = (int)((si + ei)/2)
				ei = si + 150
				np.save( save_folder+dirName[9:]+"%d.npy"%i,ln)
				i = i + 1
			
			# ln = []
			# for x in range(si, li):
			# 	[a,b,c] = arr[x]
			# 	s = a**2 + b**2 + c**2
			# 	sq = math.sqrt(s)
			# 	ln.append([a/sq, b/sq, c/sq, sq])
			# np.save(save_folder+dirName[8:]+"/"+"%d.npy"%i,ln)
		
        else:
			allFiles.append(fullPath)
        	
    return allFiles, LA, LG, LM, label_a, label_g, label_m

path1 = 'NPY_Train/'
LA, LG, LM = [], [], []
label_a, label_g, label_m = [], [], []

for fold in os.listdir(path1):
	# print path1 + fold
	x, LA, LG, LM, label_a, label_g, label_m = getListOfFiles(path1 + fold, LA, LG, LM, label_a, label_g, label_m)


# np.save('Accelerometer_Train_Data.npy',np.asarray(LA))
# np.save('Accelerometer_Train_Labels.npy',np.asarray(label_a))
print np.asarray(LA).shape, np.asarray(label_a).shape

# np.save('Gyroscope_Train_Data.npy',np.asarray(LG))
# np.save('Gyroscope_Train_Labels.npy',np.asarray(label_g))
print np.asarray(LG).shape, np.asarray(label_g).shape

# np.save('Magnetometer_Train_Data.npy',np.asarray(LM))
# np.save('Magnetometer_Train_Labels.npy',np.asarray(label_m))
print np.asarray(LM).shape, np.asarray(label_m).shape