# 
n_clusters = 10
dimension = 10
center_d = 1
scale = 0.1
size = 30
ellipticity = 5
size_range = 0

attr = 'size'
value_range = np.logspace(0, 4, 5)

dataset = synthetic.SyntheticDataSet(n_clusters,
                           dimension, 
                           center_d,
                           scale,
                           size,
                           ellipticity = ellipticity,
                           size_range=size_range)

dataset_series = synthetic.SyntheticDataSetSeries(dataset,
                                                   attr,
                                                   value_range)
dataset_series.make_series()

score_df, time_df, n_df = methods.evaluate_series(methods.clustering_methods, dataset_series)
metadata_df = methods.clustering_method_dataframe(methods.clustering_methods)

import os
dataset_series.save("src/clustering/scale")
os.makedirs("src/clustering/scale", exist_ok = True)
score_df.to_csv("src/clustering/scale/score_df.csv")
time_df.to_csv("src/clustering/scale/time_df.csv")
n_df.to_csv("src/clustering/scale/n_df.csv")
metadata_df.to_csv("src/clustering/scale/metadata_df.csv")
