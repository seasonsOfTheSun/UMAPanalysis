import synthetic
import methods

import os 
import numpy as np

#  

n_clusters = 10
dimension = 10
center_d = 1
scale = 0.1
size = 30
ellipticity = 5
size_range = 0

attr = 'scale'
value_range = np.linspace(0.05, 0.45, 11)

dataset =     synthetic.SyntheticDataSet(n_clusters,
                               dimension, 
                               center_d,
                               scale,
                               size,
                               ellipticity = ellipticity, 
                               size_range=size_range)

scale_dataset_series = synthetic.SyntheticDataSetSeries(dataset,
                                                        attr,
                                                        value_range)
scale_dataset_series.make_series()

score_df, time_df, n_df = methods.evaluate_series(methods.clustering_methods,
                        scale_dataset_series)

os.makedirs("scale", exist_ok = True)
score_df.to_csv("scale/score_df.csv")
time_df.to_csv("scale/time_df.csv")
n_df.to_csv("scale/n_df.csv")

