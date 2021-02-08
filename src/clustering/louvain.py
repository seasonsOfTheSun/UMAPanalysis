import networkx as nx
import pandas
import community
import sys
from UMAP_analysis.core import make_undirected

network_filename = sys.argv[1]
network_name = network_filename.split("/")[-1].split(".")[0]

G_dir = nx.read_gml(network_filename)
G = make_undirected(G_dir)
clustering = community.best_partition(G)
out = pandas.Series(clustering)
print(network_name)



out.to_csv(f"data/processed/clusters/louvain_{network_name}.csv")
