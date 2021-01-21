
import pandas as pd
import numpy as np
import random

import os
os.chdir('/Users/jhancock/UMAP_analysis/')

def gaussian_balls(n_clusters, points_per_cluster, n_features, noise_deviation):

    assert n_features >= n_clusters

    datapoints  = []
    for i in range(n_clusters):
        centre = np.array([j == i for j in range(n_features)])
        x = centre + noise_deviation*np.random.randn(points_per_cluster, n_features)
        df = pd.DataFrame(x)
        df["Label"] = str(i)
        datapoints.append(df)

    return pd.concat(datapoints, axis=0)

n_clusters = 5
points_per_cluster = 10
n_features = 10
noise_deviation = 0.1

print(gaussian_balls(n_clusters, points_per_cluster, n_features, noise_deviation))