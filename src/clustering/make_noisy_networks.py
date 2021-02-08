import sys
import pandas as pd
import networkx as nx
import numpy as np


from UMAP_analysis.core import umap_network, scaled_laplacian, eigenvalues


dataset = sys.argv[1]
n = int(sys.argv[2])
noise_percent = sys.argv[3]


df = pd.read_csv(f"data/intermediate/{dataset}/features.csv", index_col=0)
deviations = df.std()
scaling_factor = int(noise_percent)/100
df = df+scaling_factor*np.random.randn(*df.shape)


G = umap_network(df, n)
nx.write_gml(G,f"networks/{dataset}/nearest_neighbors_{n}_noise_percent_{noise_percent}.gml")


nodes = list(G.nodes())
transition = scaled_laplacian(G, nodes)
evecs = eigenvalues(transition, nodes)
evecs.to_csv( f"networks/{dataset}/eigenvectors/nearest_neighbors_{n}_noise_percent_{noise_percent}.gml")
