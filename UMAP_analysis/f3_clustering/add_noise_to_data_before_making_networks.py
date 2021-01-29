import umap
import numba
import numpy as np
import pandas as pd
import networkx as nx


from UMAP_analysis.f2_build_networks.make_network import umap_network
"""
n = 14
for dataset in ['lish-moa']:
    df = pd.read_csv(f"munged_data/{dataset}/features.csv", index_col=0)
    deviations = df.std()
    for scaling_factor in np.linspace(0,1,9):
        G = umap_network(df+scaling_factor*deviations, n)
        nx.write_gml(G, f"UMAP_analysis/f3_clustering/noised_networks/{dataset}/similarity_with_noise_level_{scaling_factor}.gml")
"""

n = 14
for dataset in ['lish-moa']:
    df = pd.read_csv(f"munged_data/{dataset}/features.csv", index_col=0)
    for scaling_factor in np.linspace(0,1,9):
        deviations = df.std()*np.random.randn(*noise_df.shape)
        noise_df = df+scaling_factor*deviations
        filename = f"UMAP_analysis/f3_clustering/cluster/{dataset}/similarity_with_noise_level_{scaling_factor}.g\
ml"
        
