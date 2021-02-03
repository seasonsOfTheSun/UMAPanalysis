import networkx as nx
import community
import sys


network_filename = sys.argv[1]


G = nx.read_gml(network_filename)
clustering = community.best_partition(G)
tmp = pandas.Series(clustering)
tmp.name = scaling_factor
out.append(tmp)


df = pandas.concat(out, axis = 1)
df.index.name = "scaling_factor"
df.to_csv(f"UMAP_analysis/data/processed/clusters/louvain_{dataset}_{noise_level}.csv)
