import pdb

import sklearn.ensemble as se
import pandas as pd
import time

os.chdir('/Users/jhancock/UMAP_analysis/')
dataset = 'l1000'
pd.concat(out, axis = 1).to_csv(f"4.classification/predictions/{dataset}/random_forest_predictions.csv")
pd.Series(out_time, name="Time", index=classes.columns).to_csv("performance_metrics/random_forest_prediction_times.csv")
