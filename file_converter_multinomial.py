import sys
sys.path.insert(1, '/Users/reiota/Desktop/conda/anaconda3/lib/python3.12/site-packages')
import numpy as np
import pandas as pd
import scipy.stats as stats
from dataclasses import dataclass
from typing import Dict, Tuple

FACTORS = 100
GENES = 20000
ETA = 25
TARGET_STRENGTH = 35

df = pd.read_feather('data.feather')
df['background effect'] = (np.random.random(size = len(df))-0.5)
#df['p_hat'] = np.divide(np.ones(len(df)), np.add(np.ones(len(df)), np.exp(-df['background effect'])))
column_names = [f"x{i}" for i in range(GENES)]
df_new = df[column_names]
df_new.replace({0:'q1', 1:'q1', 2:'q2', 3:'q2', 4:'q3', 5:'q3', 6:'q4', 7:'q4'}, inplace=True)
dummy_val_df = pd.get_dummies(df_new, columns=column_names, drop_first=True)
factors = dummy_val_df.sample(FACTORS, axis=1)
x_frequency = factors.sum()/len(df)
#strength = (np.random.random(size=FACTORS)-0.5)*ETA/FACTORS
strength = (np.random.random(size=FACTORS)-0.5)
total_strength = sum(abs(strength))
strength = strength / total_strength * TARGET_STRENGTH
x_contribution = x_frequency.mul(strength)
average_contribution = x_contribution.sum()
#print(average_contribution)
factors = factors.mul(strength)
factors["gene contribution"] = factors.apply(np.sum, axis=1)
factors["total"] = np.add(factors["gene contribution"], df['background effect'])
new_column_names = [f"x{i}" for i in range(GENES*3)]
dummy_val_df = dummy_val_df.set_axis(new_column_names, axis=1)
dummy_val_df["y"] = np.random.binomial(1, np.divide(np.ones(len(dummy_val_df)),
                 np.add(np.ones(len(dummy_val_df)), np.exp(-factors['total']))), size = len(dummy_val_df))
dummy_val_df["background effect"] = df["background effect"] + average_contribution
dummy_val_df["p_hat"] = np.divide(np.ones(len(df)), np.add(np.ones(len(df)), np.exp(-dummy_val_df['background effect'])))
dummy_val_df.to_feather('binary_20k_v2_18.feather')