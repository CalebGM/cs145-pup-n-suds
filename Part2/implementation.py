import pandas as pd
import numpy as np
from math import atan2

df = pd.read_csv(r'../data.csv')

#features
#cartesian to polar conversion
divZero = df['loc_x'] == 0
df['r'] = np.sqrt(df['loc_x']**2 + df['loc_y']**2)
df['theta'] = np.zeros(len(df))
df['theta'][~divZero]  = np.arctan2(df['loc_y'][~divZero],df['loc_x'][~divZero])
df['theta'][divZero] = np.pi/2 

#time
df['time'] = df['minutes_remaining']*60 + df['seconds_remaining']