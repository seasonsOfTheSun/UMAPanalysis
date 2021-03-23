import sys
import re
import networkx as nx
import pandas as pd
import numpy as np
from UMAP_analysis.core import make_undirected
import matplotlib.pyplot as plt


path = sys.argv[1]
m = re.match(".*/networks/(?P<dataset>.*?)/(?P<name>.*?).gml", path)
dataset = m.groupdict()['dataset']
name = m.groupdict()['name']


G = nx.read_gml(path)
G = make_undirected(G)
ev = pd.read_csv(f"../networks/{dataset}/eigenvectors/{name}.csv", index_col = 0)


gc = max(nx.connected_components(G), key = lambda x:len(x))
gc = list(gc)

e_1 = sys.argv[2]
e_2 = sys.argv[3]
pos = {i:(ev.loc[i,e_1],ev.loc[i,e_2]) for i in G.nodes()}
xy = nx.spring_layout(G, pos=pos)


df = pd.DataFrame([xy[i] for i in i in G.nodes()]) 
df.to_csv(f"../networks/{dataset}/postitions/{name}.csv")
