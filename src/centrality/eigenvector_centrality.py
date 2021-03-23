import networkx as nx
import pandas as pd
import sys
import re


network_filename = sys.argv[1]
m = re.match("networks/(?P<dataset>.*?)/(?P<name>.*?).gml", network_filename)
dataset = m.groupdict()['dataset']
name = m.groupdict()['name']


G = nx.read_gml(network_filename)
centralities = pd.Series(nx.eigenvector_centrality(G))
centralities.name = name

centralities.to_csv(f"data/processed/predictions/eigenvector_centralities/{dataset}/{name}.csv")
