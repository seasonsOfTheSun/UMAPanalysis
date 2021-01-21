

import pandas as pd
import scipy.stats
import scipy.sparse

centralities = pd.read_csv(f"5.centrality/eigencentrality/{dataset}/.csv", index_col = 0, header = None)[1]
centralities.name = 'centrality'
scipy.stats.mannwhitneyu(centralities[metadata.cp_type == 'trt_cp'], centralities[metadata.cp_type != 'trt_cp'])

nodes = G.nodes()
labels = pd.read_csv(f"munged_data/{dataset}/labels.csv", index_col = 0)
metadata = pd.read_csv(f"munged_data/{dataset}/metadata_matrix.csv", index_col = 0)

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.boxplot([centralities[metadata.cp_type == 'trt_cp'],
            centralities[metadata.cp_type != 'trt_cp']])
ax.set_xticklabels(["Compound", "DMSO"])
ax.set_ylabel("Eigencentrality")

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.hist(centralities[metadata.cp_type == 'trt_cp'],density= True,log= True, bins = 200)
ax.hist(centralities[metadata.cp_type != 'trt_cp'],density= True,log= True, bins = 4)
ax.set_ylabel("Eigencentrality")