import pandas as pd
import numpy as np


df = pd.read_csv(r'../data.csv')
shot_attempt = df.shot_made_flag
list = []

for index, shot in enumerate(shot_attempt):
	if np.isnan(shot):
		list.append(index)
	
	
nanIndex = pd.DataFrame(list, columns=['col1'])
shotsMade = shot_attempt[shot_attempt == 1]
shotsMissed = shot_attempt[shot_attempt == 0]
numMade = len(shotsMade)
numMissed = len(shotsMissed)
numLost = len(nanIndex)
total = numMade + numMissed
print("Number of shots made: ", numMade)
print("Number of shots missed: ", numMissed)
print("Number of missing values: ", numLost)
print("Percentage of values missing: ",1. * numLost/len(shot_attempt),"%")
print("Percentage of shots made: ",1. * numMade/total,"%")
print("Percentage of shots missed: ",1. * numMissed/total,"%")
