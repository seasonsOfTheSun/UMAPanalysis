import networkx as nx
import numpy as np
import pandas as pd
import sys

cluster_name = sys.arvg[1]
cluster = pd.read_csv(cluster_name)
cluster = [set(cluster[cluster==i].index) for i in cluster.unique()]

network_name = sys.argv[2]

G = nx.read_gml(network_name)

out = []
for community in cluster:
    L_c = sum(wt for u, v, wt in G.edges(comm, data=weight, default=1) if v in comm)
    out_degree_sum = sum(out_degree[u] for u in comm)
    in_degree_sum = sum(in_degree[u] for u in comm) if directed else out_degree_sum
    out.append(L_c / m - out_degree_sum * in_degree_sum * norm)


