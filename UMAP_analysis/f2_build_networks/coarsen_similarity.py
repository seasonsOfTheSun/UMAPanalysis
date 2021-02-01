import networkx as nx
import pandas as pd
import scipy.sparse
import matplotlib.pyplot as plt

import os
condensed_moas.columns = moas.columns
condensed_moas

G_nx = nx.read_gml("networks/l1000/similarity.gml")
nx.write_gml(Gc, "networks/l1000/coarsened_similarity.gml")
scipy.sparse.save_npz("networks/l1000/similarity_coarsening_map.npz", C)
moas = pd.read_csv("munged_data/moas.csv", index_col = 0)
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




