
import sys
import os
import pandas as pd
import networkx as nx
import numpy as np
import time

from UMAP_analysis.core import umap_network


dataset = sys.argv[1]
n = int(sys.argv[2])
metric = sys.argv[3]

df = pd.read_csv(f"data/intermediate/{dataset}/features.csv", index_col=0)

# iteration = str(np.random.randint(10**16-1)).rjust(16, '0')

start_time = time.time()
G = umap_network(df, n, metric = metric)
G = G.to_undirected()
end_time = time.time()
evaluation_time = end_time-start_time

os.makedirs(f"networks/{dataset}/", exist_ok = True)

nx.write_gml(G,f"networks/{dataset}/metric_{metric}_nneighbors_{n}-{iteration}.gml")
fp = open(f"networks/{dataset}/evaluation_time_metric_{metric}_nneighbors_{n}.gml", 'w')
fp.write(str(evaluation_time))
fp.close()
