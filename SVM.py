import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn import svm

df = pd.read_csv(r'./data.csv')

#features
#shot_distance is less precise
divZero = df['loc_x'] == 0
df['r'] = np.sqrt(df['loc_x']**2 + df['loc_y']**2)
df['theta'] = np.zeros(len(df))
df['theta'][~divZero]  = np.arctan2(df['loc_y'][~divZero],df['loc_x'][~divZero])
df['theta'][divZero] = np.pi/2 
df['seconds_from_period_end']=60*df['minutes_remaining']+df['seconds_remaining']

removes = ['combined_shot_type', 'game_event_id', 'game_id', 'lat', 'loc_x',\
          'loc_y', 'lon', 'minutes_remaining', 'seconds_remaining',\
          'shot_distance', 'shot_zone_area', 'shot_zone_basic',\
          'shot_zone_range', 'team_id', 'team_name', 'game_date',\
          'matchup', 'shot_id'],
for remove in removes:
    df = df.drop(remove, 1)

dummies = ['action_type', 'shot_type', 'opponent', 'period', 'season']
for dummy in dummies:
    df = pd.concat([df, pd.get_dummies(df[dummy], prefix=dummy)], 1)
    df = df.drop(dummy, 1)
    
#SVM
train = df[pd.notnull(df['shot_made_flag'])]
X = train.drop('shot_made_flag', 1)
y = train['shot_made_flag']
X = scale(X)
svc = svm.LinearSVC()
svc.fit(X, y)
score = svc.score(X, y)

to_predict = df[pd.isnull(df['shot_made_flag'])]
to_predict = to_predict.drop('shot_made_flag', 1)
predicted = svc.predict(to_predict)