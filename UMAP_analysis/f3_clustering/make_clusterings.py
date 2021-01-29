import networkx
import community
import pandas
import numpy
import os
import re

dataset = 'lish-moa'
filenames = os.listdir(f"UMAP_analysis/f3_clustering/noised_networks/{dataset}/")
scaling_factors  = [re.match('similarity_with_noise_level_(?P<x>.*?).gml',i).groupdict()['x'] for i in filenames]

out = []
for scaling_factor in scaling_factors:
    print(scaling_factor)
    G = networkx.read_gml(f"UMAP_analysis/f3_clustering/noised_networks/{dataset}/similarity_with_noise_level_{scaling_factor}.gml")
    G_u = G.to_undirected()
    for edge in G_u.edges():
        i,j = edge

        try:
            w = G[(j,i)]['weight']
            G_u[(i,j)]['weight'] += w

        except KeyError:
            pass
"""

    clustering = community.best_partition(G_u)
    tmp = pandas.Series(clustering)
    tmp.name = scaling_factor
    out.append(tmp)


df = pandas.concat(out, axis = 1)
df.index.name = "scaling_factor"
df.to_csv(f"UMAP_analysis/f3_clustering/clusters/UMAP_{dataset}.csv")
"""


def umap_network(df, nn):
    rndstate = np.random.RandomState()
    knn_net = umap.umap_.fuzzy_simplicial_set(df.values, nn, rndstate, 'manhattan')
    G = nx.from_numpy_array(knn_net[0].toarray(), create_using = nx.DiGraph)
    num_to_id = dict(enumerate(df.index))
    return nx.relabel_nodes(G, num_to_id.get)
git add 
