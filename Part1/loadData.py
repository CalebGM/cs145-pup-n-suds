import pandas as pd
import numpy as np


df = pd.read_csv(r'../data.csv')
shot_attempt = df.shot_made_flag
for index, shot in enumerate(shot_attempt):
	if np.isnan(shot):
		print(index)