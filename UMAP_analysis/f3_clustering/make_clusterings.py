import networkx
import community
import numpy
import pandas
import os
import re

dataset = 'cytodata'
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


    clustering = community.best_partition(G_u)
    tmp = pandas.Series(clustering)
    tmp.name = scaling_factor
    out.append(tmp)


df = pandas.concat(out, axis = 1)
df.index.name = "scaling_factor"
df.to_csv(f"UMAP_analysis/f3_clustering/clusters/UMAP_{dataset}.csv")
