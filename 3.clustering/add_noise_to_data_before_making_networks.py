import pandas as pd

import umap

import numpy as np
import numba
import networkx as nx

import os
os.chdir('/Users/jhancock/UMAP_analysis/')


def umap_network(df, nn):
    rndstate = np.random.RandomState()
    knn_net = umap.umap_.fuzzy_simplicial_set(df.values, nn, rndstate, 'manhattan')
    G = nx.from_numpy_array(knn_net[0].toarray(), create_using = nx.DiGraph)
    num_to_id = dict(enumerate(df.index))
    return nx.relabel_nodes(G, num_to_id.get)

import os
os.chdir('//')


n = 14
for dataset in ['GDSC']:
    print()
    df = pd.read_csv(f"munged_data/{dataset}/features.csv", index_col=0)
    deviations = df.std()
    for scaling_factor in np.linspace(0,2,5):
        G = umap_network(df+scaling_factor*deviations, n)
        nx.write_gml(G, f"3.clustering/noised_datasets/{dataset}/similarity_with_noise_level_{str(scaling_factor)}.gml")
