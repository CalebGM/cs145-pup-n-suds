import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn import svm
pd.options.mode.chained_assignment = None  # default='warn'

df = pd.read_csv(r'./data.csv')

shot_attempt = df.shot_made_flag
shotsMade = shot_attempt[shot_attempt == 1]
shotsMissed = shot_attempt[shot_attempt == 0]
numMade = len(shotsMade)
numMissed = len(shotsMissed)
nan_rows = df[pd.isnull(df['shot_made_flag'])]
nan_ids = nan_rows.shot_id

#features
#conversion from cartesian to polar; shot_distance less precise
#also seperating angle/distance which is more "intuitive" for difficulty?
divZero = df['loc_x'] == 0
df['r'] = np.sqrt(df['loc_x']**2 + df['loc_y']**2)
df['theta'] = np.zeros(len(df))
df.loc[~divZero, 'theta']  = np.arctan2(df['loc_y'][~divZero],df['loc_x'][~divZero])
df.loc[divZero, 'theta'] = np.pi/2 
#summing minutes and seconds to get seconds
df['seconds_from_period_end']=60*df['minutes_remaining']+df['seconds_remaining']

#features which are either the same for all, or directly related to others
removes = ['combined_shot_type', 'game_event_id', 'game_id', 'lat', 'loc_x',\
          'loc_y', 'lon', 'minutes_remaining', 'seconds_remaining',\
          'shot_distance', 'shot_zone_area', 'shot_zone_basic',\
          'shot_zone_range', 'team_id', 'team_name', 'game_date',\
          'matchup', 'shot_id'],
for remove in removes:
    df = df.drop(remove, 1)

#one-hot encoding for categorical features
dummies = ['action_type', 'shot_type', 'opponent', 'period', 'season']
for dummy in dummies:
    df = pd.concat([df, pd.get_dummies(df[dummy], prefix=dummy)], 1)
    df = df.drop(dummy, 1)
    
#SVM
train = df[pd.notnull(df['shot_made_flag'])]
X = train.drop('shot_made_flag', 1)
y = train['shot_made_flag']
#scale data for svm
X = scale(X)
svc = svm.LinearSVC()
svc.fit(X, y)
score = svc.score(X, y)

to_predict = df[pd.isnull(df['shot_made_flag'])]
to_predict = to_predict.drop('shot_made_flag', 1)
#scale data for svm
to_predict = scale(to_predict)
predicted = svc.predict(to_predict)

d = {'shot_id' : nan_ids,
	'shot_made_flag' : predicted}
results = pd.DataFrame(d);

results.to_csv('resultsSVM.csv', index=False)
print (score)
print (predicted[0:9])
print (float(sum(predicted))/len(predicted))
print (float(numMade)/(numMade+numMissed))