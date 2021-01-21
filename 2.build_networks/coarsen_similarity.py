import networkx as nx
import pandas as pd
import scipy.sparse
import pygsp
import graph_coarsening as gc
import matplotlib.pyplot as plt

import os

G_nx = nx.read_gml("networks/l1000/similarity.gml")
G_nx = max(nx.weakly_connected_component_subgraphs(G_nx), key = lambda x:len(x.nodes()))
nodes = sorted(list(G_nx.nodes()))
G = pygsp.graphs.Graph(nx.adjacency_matrix(G_nx, nodelist=nodes))

C,Gc,*_ = gc.coarsen(G, r = 0.99)
Gc = nx.from_scipy_sparse_matrix(Gc.W)
pos = nx.spring_layout(Gc)
Gc_nodes = list(sorted(Gc.nodes()))

nx.write_gml(Gc, "networks/l1000/coarsened_similarity.gml")
scipy.sparse.save_npz("networks/l1000/similarity_coarsening_map.npz", C)

moas = pd.read_csv("munged_data/moas.csv", index_col = 0)
moas = moas.reindex(nodes)
moa_matrix = scipy.sparse.csr_matrix(moas.values)

condensed_moas = pd.DataFrame((C * moa_matrix).toarray())
condensed_moas.columns = moas.columns
condensed_moas

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

"""
nx.draw_networkx(Gc,
                 ax=ax,
                 pos=pos,
                 nodelist=Gc_nodes,
                 node_size=1,
                 with_labels=False,
                 node_color=condensed_moas.loc[Gc_nodes, 'tubulin_inhibitor'],
                 cmap='viridis'
                 )
"""




