import networkx as nx
import pandas as pd
import sys
import re

from UMAP_analysis.core import nearest_neighbor

filename = sys.argv[1]
m = re.match("networks/(?P<dataset>.*?)_similarity.*?.gml", filename)
dataset = m.groupdict()['dataset']
G = nx.read_gml(filename)

true_labels = pd.read_csv(f"data/intermediate/{dataset}/labels.csv", index_col = 0)
train = pd.read_csv(f"data/intermediate/{dataset}/train.csv", header = None)[0].values
train = set(train)

mask = [i in train for i in true_labels.index]
mask = pd.Series(mask, index = true_labels.index)
labels = true_labels.mask(~mask).fillna(0)

out = []
for moa in labels.columns:
    predictions = nearest_neighbor(G, labels[moa])
    out.append(predictions)
    break
pd.concat(out, axis = 1).to_csv("data/intermediate/predictions/"+filename.split("/")[-1].split(".")[0]+".csv")
