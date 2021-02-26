import sys
import pandas as pd
import networkx as nx
import numpy as np


from UMAP_analysis.core import umap_network


dataset = sys.argv[1]
n = int(sys.argv[2])
metric = sys.argv[3]
noise_percent = sys.argv[4]


df = pd.read_csv(f"data/intermediate/{dataset}/features.csv", index_col=0)
scaling_factor = int(noise_percent)/100
df = df+scaling_factor*np.random.randn(*df.shape)


iteration = str(np.random.randint(10**16-1)).rjust(16, '0')
G = umap_network(df, n)
nx.write_gml(G,f"networks/{dataset}/metric_{metric}_nneighbors_{n}_noise_percent_{noise_percent}-{iteration}.gml")
