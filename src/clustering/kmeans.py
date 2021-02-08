import sys
import pandas as pd
import networkx as nx
import numpy as np
from UMAP_analysis.core import umap_network, scaled_laplacian, eigenvalues


dataset = sys.argv[1]
noise_percent = sys.argv[2]


df = pd.read_csv(f"data/intermediate/{dataset}/features.csv", index_col=0)
deviations = df.std()
scaling_factor = int(noise_percent)/100
df = df+scaling_factor*np.random.randn(*df.shape)


import sklearn.cluster
_,clusters,_ = sklearn.cluster.k_means(df,2)
clusters = pd.Series(clusters, index = df.index)
clusters.to_csv(f"data/processed/clusters/{dataset}/kmeans_noise_percent_{noise_percent}.csv")
