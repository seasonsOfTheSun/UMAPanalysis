
import networkx as nx
import pandas as pd
import sys
import re
import numpy as np
from UMAP_analysis.core import nearest_neighbor

filename = sys.argv[1]
m = re.match("networks/(?P<dataset>.*?)/(?P<name>.*?)\.gml", filename)
dataset = m.groupdict()['dataset']
name = m.groupdict()['name']
G = nx.read_gml(filename)

true_labels = pd.read_csv(f"data/intermediate/{dataset}/labels.csv", index_col = 0)
train = pd.read_csv(f"data/intermediate/{dataset}/train.csv", header = None)[0].values
test = pd.read_csv(f"data/intermediate/{dataset}/test.csv", header = None)[0].values

not_masked = [i in train for i in true_labels.index]
not_masked = pd.Series(not_masked, index = true_labels.index)
labels = true_labels.mask(~not_masked).fillna(0)

out = []
for moa in labels.columns:
    predictions = pd.Series(nearest_neighbor(G, labels[moa]))
    out.append(predictions)

iteration = str(np.random.randint(10**16-1)).rjust(16, '0')
df = pd.concat(out, axis = 1)
df = df.loc[test]
df.columns = labels.columns
df.to_csv(f"data/processed/predictions/{dataset}/nearest_neighbors_{name}_training_set_1.csv")
