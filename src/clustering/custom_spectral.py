import re
import networkx as nx
import sys
import pandas as pd
from UMAP_analysis.core import scaled_laplacian_opposite, eigenvalues

filename = sys.argv[1]
n_clusters = 40
n_dims = 10
expr = "networks/(?P<dataset>.*?)/noisy/(?P<name>.*?).gml"
m = re.match(expr, filename)
dataset = m.groupdict()['dataset']
name = m.groupdict()['name']

G = nx.read_gml(filename)
nodes = list(G.nodes())



laplacian = scaled_laplacian_opposite(G, nodes)
evecs = eigenvalues(laplacian, nodes)


import sklearn.cluster
_,clusters,_ = sklearn.cluster.k_means(evecs.iloc[:-1,:n_dims+1],n_clusters)
clusters = pd.Series(clusters, index = evecs.index[:-1])
clusters.to_csv(f"data/processed/clusters/{dataset}/custom_spectral_n_clusters_{n_clusters}_{name}.csv", header = None)
