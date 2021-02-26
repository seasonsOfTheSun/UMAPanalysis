
import sys
import pandas as pd
import networkx as nx
import numpy as np

from UMAP_analysis.core import umap_network, scaled_transition, eigenvalues, largest_connected_component


dataset = sys.argv[1]
n = int(sys.argv[2])
metric = sys.argv[3]

df = pd.read_csv(f"data/intermediate/{dataset}/features.csv", index_col=0)

iteration = str(np.random.randint(10**16-1)).rjust(16, '0')
G = umap_network(df, n, metric = metric)
nx.write_gml(G,f"networks/{dataset}/metric_{metric}_nneighors_{n}-{iteration}.gml")
