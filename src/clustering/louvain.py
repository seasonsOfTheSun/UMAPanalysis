import networkx as nx
import pandas
import community
import sys
import re
from UMAP_analysis.core import make_undirected


network_filename = sys.argv[1]
m = re.match("networks/(?P<dataset>.*?)/noisy/(?P<name>.*?).gml", network_filename)
dataset = m.groupdict()['dataset']
name = m.groupdict()['name']


G_dir = nx.read_gml(network_filename)
G = make_undirected(G_dir)
clustering = community.best_partition(G)
out = pandas.Series(clustering)


out.to_csv(f"data/processed/clusters/{dataset}/louvain_{name}.csv", header = None)
