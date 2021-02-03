import networkx as nx
import sys
import re

from UMAP_analysis.core import nearest_neighbors

filename = sys.argv[1]
m = re.match("networks/{dataset}_similarity.*?.gml", filename)
dataset = m.groupdict()['dataset']
G = nx.read_gml(filename)

true_labels = pd.read_csv("data/intermediate/{dataset}/labels.csv", index_col = 0)
train = pd.read_csv("data/intermediate/{dataset}/train.csv",index_col = 0)
train = train.reindex(true_labels.index)


out = []
for moa in labels.columns:
    predictions = nearest_neighbor(G, labels[moa])
    out.append(predictions)
pd.concat(out, axis = 1).to_csv("data/intermediate/predictions/"+filename.split("/")[-1].split(".")[0]+")
