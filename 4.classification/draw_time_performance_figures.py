import pandas as pd
import numpy as np

df = pd.read_csv("performance_metrics/auc_for_prediction_methods.csv", index_col = 0)

nearest_neighbor_times = pd.read_csv("performance_metrics/nearest_neighbor_prediction_times.csv", index_col = 0, header = None)
propagation_times = pd.read_csv("performance_metrics/propagation_prediction_times_nn_equal_4.csv", index_col = 0, header = None)
random_forest_times = pd.read_csv("performance_metrics/random_forest_prediction_times.csv", index_col = 0,header = None)

# comment out if you fix this stupid bug
df = -df

df['PropagationTimes'] = propagation_times[1]
# "black"
df['RandomForestTimes'] = random_forest_times[1]
#'#00c2d7'
df['NearestNeighborTimes'] = nearest_neighbor_times[1]
#'#ae342b'

import matplotlib.pyplot as plt


fig = plt.figure(figsize=(10,5))
ax = fig.add_axes([0.1,0.1,0.8,0.8])
ax.scatter(x=df.Propagation, y=df.PropagationTimes, c = 'k')
ax.scatter(x=df.RandomForest, y=df.RandomForestTimes, c = '#00c2d7')
ax.scatter(x=df.NearestNeighbor, y=df.NearestNeighborTimes, c = '#ae342b')
ax.set_xlabel("AUROC")
ax.set_yscale("log")
ax.set_ylabel("Runtime (seconds)")

fig.show()
fig.savefig("figures/classification_performance.svg")