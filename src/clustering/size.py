import synthetic
import methods
import numpy as np

n_clusters = 10
dimension = 10
center_d = 1
scale = 0.1
size = 30
ellipticity = 5
size_range = 0

attr = 'size'
value_range = np.logspace(1, 5, 5)

transform_dataset = """amplitude = 1
period = 10
n = 100

for i in range(100):
    col = np.random.choice(self.original_features, n)
    randmat = np.random.randn(n)
    bart = (randmat.reshape((1,n)) * self.data[col]).sum(axis = 1)

    self.data[f"Transformed_{i}"] = list(map(lambda x : x**2, bart))

for col in self.original_features:
    del self.data[col]

self.data = self.data/self.data.var()
true_dimension = len(self.data.columns)
"""

dataset = synthetic.SyntheticDataSet(n_clusters,
                           dimension, 
                           center_d,
                           scale,
                           size,
                           ellipticity = ellipticity,
                           size_range=size_range,
                           transform_dataset = transform_dataset)

dataset_series = synthetic.SyntheticDataSetSeries(dataset,
                                                   attr,
                                                   value_range)
dataset_series.make_series()

score_df, time_df, n_df = methods.evaluate_series(methods.clustering_methods, dataset_series)
metadata_df = methods.clustering_method_dataframe(methods.clustering_methods)
metadata_df.index.name = "clustering methods"

import os
dataset_series.save("src/clustering/scale")
os.makedirs("src/clustering/scale", exist_ok = True)
score_df.to_csv("src/clustering/scale/score_df.csv")
time_df.to_csv("src/clustering/scale/time_df.csv")
n_df.to_csv("src/clustering/scale/n_df.csv")
metadata_df.to_csv("src/clustering/scale/metadata_df.csv")
