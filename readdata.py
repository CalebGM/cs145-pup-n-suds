import pandas as pd
import numpy as np
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.optics import optics 
from sklearn.neighbors import NearestNeighbors


df = pd.read_csv(r'./data.csv')
shot_attempt = df.shot_made_flag
list_1 = []

for index, shot in enumerate(shot_attempt):
    if np.isnan(shot):
        list_1.append(index)


nanIndex = pd.DataFrame(list_1, columns=['col1'])
shotsMade = shot_attempt[shot_attempt == 1]
shotsMissed = shot_attempt[shot_attempt == 0]
numMade = len(shotsMade)
numMissed = len(shotsMissed)
numLost = len(nanIndex)
total = len(shot_attempt)
'''print("Number of shots made: ", numMade)
print("Number of shots missed: ", numMissed)
print("Number of missing values: ", numLost)
print("Percentage of values missing: ",float(numLost)/total,"%")
print("Percentage of shots made: ",float(numMade)/total,"%")
print("Percentage of shots missed: ",float(numMissed)/total,"%")'''
new_cord =df[['loc_x','loc_y','action_type','combined_shot_type','minutes_remaining','seconds_remaining','shot_zone_area','shot_zone_basic','shot_distance']]
shotZoneList= df.shot_zone_area.values
shotZoneBasicList=df.shot_zone_basic.values
shotZones =np.unique(shotZoneList)
shotZoneBasics =np.unique(shotZoneBasicList)
'''actionTypeList= df.action_type.values
combinedShotTypeList=df.combined_shot_type.values
actionTypes=np.unique(actionTypeList)
combinedShotTypes=np.unique(combinedShotTypeList)
for index_1, actionType in enumerate(actionTypes):
    actionTypeList[actionTypeList==actionType]=index_1
for index_2, combinedShotType in enumerate(combinedShotTypes):	     
    combinedShotTypeList[combinedShotTypeList==combinedShotType]=index_2'''
for index_1, shotZone in enumerate(shotZones):
    shotZoneList[shotZoneList==shotZone]=index_1
for index_2, shotZoneBasic in enumerate(shotZoneBasics):	     
    shotZoneBasicList[shotZoneBasicList==shotZoneBasic]=index_2
divZero = new_cord['loc_x'] == 0
'''new_cord['r'] = np.sqrt(new_cord['loc_x']**2 + new_cord['loc_y']**2)'''
new_cord['theta'] = np.zeros(len(new_cord))
new_cord['theta'][~divZero]  = np.arctan2(new_cord['loc_y'][~divZero],new_cord['loc_x'][~divZero])
new_cord['theta'][divZero] = np.pi/2 
new_cord['scaled_shot_distance']=new_cord['shot_distance']/30.0

xMax=max(new_cord["loc_x"])
xMin=min(new_cord["loc_x"])
yMax=max(new_cord['loc_y'])
yMin=min(new_cord['loc_y'])
new_cord['scaled_x']=new_cord['loc_x']/float(xMax-xMin)
new_cord['scaled_y']=new_cord['loc_y']/float(yMax-yMin)
new_cord['seconds_from_period_end']=60*new_cord['minutes_remaining']+new_cord['seconds_remaining']
new_cord['onset_turn']=new_cord['seconds_from_period_end']/24.0
new_cord.drop('action_type',axis=1,inplace=True)
new_cord.drop('combined_shot_type',axis=1,inplace=True)
new_cord.drop('seconds_remaining',axis=1,inplace=True)
new_cord.drop('minutes_remaining',axis=1,inplace=True)
new_cord.drop('seconds_from_period_end',axis=1,inplace=True)
new_cord.drop('loc_x',axis=1,inplace=True)
new_cord.drop('loc_y',axis=1,inplace=True)
new_cord.drop('shot_zone_area',axis=1,inplace=True)
new_cord.drop('shot_zone_basic',axis=1,inplace=True)
new_cord.drop('shot_distance',axis=1,inplace=True)
featuresImp=new_cord.values
'''featuresImp=np.c_[featuresImp,actionTypeList]
featuresImp=np.c_[featuresImp,combinedShotTypeList]'''
'''featuresImp=np.c_[featuresImp,shotZoneList]
featuresImp=np.c_[featuresImp,shotZoneBasicList]'''
#choose k neighbors
kNeighbors=5
nbrs= NearestNeighbors(n_neighbors=kNeighbors,algorithm='ball_tree').fit(featuresImp)
distances, indices= nbrs.kneighbors()
lostData=indices[list_1]
neighborStat=[]
for points in lostData:
    k=0
    for neigh in points:    
        if (shot_attempt[neigh]==1):
            k=k+1
    neighborStat.append(k)
lostPredict=np.greater_equal(neighborStat,3)
lostPredict=lostPredict*1
print (float(sum(lostPredict))/len(lostPredict))
#the accuracy calculated from known values
print (float(numMade)/(numMade+numMissed))
