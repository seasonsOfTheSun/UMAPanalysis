import umap
import numba
import numpy as np
import pandas as pd
import networkx as nx

from UMAP_analysis.f2_build_networks.make_networks import umap_network

n = 14
for dataset in ['toy']:
    df = pd.read_csv(f"munged_data/{dataset}/features.csv", index_col=0)
    deviations = df.std()
    for scaling_factor in np.linspace(0,2,5):
        G = umap_network(df+scaling_factor*deviations, n)
        pdb.set_trace()
        nx.write_gml(G, f"f3_clustering/noised_datasets/{dataset}/similarity_with_noise_level_{scaling_factor}.gml")

