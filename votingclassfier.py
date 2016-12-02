import pandas as pd
import numpy as np
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.optics import optics 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.ensemble import VotingClassifier

df = pd.read_csv(r'../data.csv')
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
data_cl=df.copy()
divZero= df['loc_x']==0
df['theta'] = np.zeros(len(df))
df['theta'][~divZero]  = np.arctan2(df['loc_y'][~divZero],df['loc_x'][~divZero])
df['theta'][divZero] = np.pi/2 
df['seconds_from_period_end']=60*df['minutes_remaining']+df['seconds_remaining']

removes = ['action_type', 'shot_type', 'opponent', 'period', 'season',\
'combined_shot_type', 'game_event_id', 'game_id', 'lat',\
'lon', 'minutes_remaining', 'seconds_remaining',\
'shot_zone_area', 'shot_zone_basic',\
'shot_zone_range', 'team_id', 'team_name', 'game_date',\
'matchup', 'shot_id','shot_made_flag'],
for remove in removes:
    df = df.drop(remove, 1)
#scale data for clustering	
df=scale(df)
#convert string data to existence value	
dummies = ['action_type', 'shot_type', 'opponent', 'shot_zone_area','shot_zone_basic']
for dummy in dummies:
    df= pd.concat([pd.DataFrame(df), pd.DataFrame(pd.get_dummies(data_cl[dummy], prefix=dummy))],axis=1)

train_set= df[pd.notnull(data_cl['shot_made_flag'])]
train_class= shot_attempt[pd.notnull(data_cl['shot_made_flag'])]	
'''
knn_grid= GridSearchCV(
          estimator=KNeighborsClassifier()
		  ,
		  param_grid={
		  'n_neighbors':[5,9,15,25],
		  'algorithm':['ball_tree'],
		  'leaf_size':[2,3,4],
		  'p':[1]
		  },
		  cv=5,
		  )
knn_grid.fit(train_set,train_class)

print(knn_grid.best_score_)
print(knn_grid.best_params_)

'''
#neighbors model
knn_model=KNeighborsClassifier(n_neighbors=25,algorithm='ball_tree',leaf_size=3,p=1)
#svm model
svm_model=svm.SVC(kernel='rbf',probability=True)
#voting classifier with weight=[2,1] for neighbors and svm respectively
eclf=VotingClassifier(estimators=[('knn',knn_model),('svm',svm_model)],voting='soft',weights=[2,1])
eclf.fit(train_set,train_class)
test_set=df[pd.isnull(data_cl['shot_made_flag'])]
predicted=eclf.predict(test_set)
#the accuracy calculated from known values

print float(sum(predicted))/len(predicted)
print float(numMade)/(numMade+numMissed)
