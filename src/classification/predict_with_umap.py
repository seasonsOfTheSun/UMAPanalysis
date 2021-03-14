import sys
import re
import pandas as pd
import networkx as nx
from UMAP_analysis.core import scaled_transition, largest_connected_component, propagate

network_filename = sys.argv[1]
m = re.match("networks/(?P<dataset>.*?)/(?P<name>.*?).gml", network_filename)
dataset = m.groupdict()['dataset']
name = m.groupdict()['name']

G=nx.read_gml(network_filename)
labels=pd.read_csv(f"data/intermediate/{dataset}/labels.csv", index_col=0)
metadata = pd.read_csv(f"data/intermediate/{dataset}/metadata.csv", index_col=0)
features = pd.read_csv(f"data/intermediate/{dataset}/features.csv", index_col=0)



train = pd.read_csv(f"data/intermediate/{dataset}/train.csv", header = None)[0].values
test  = pd.read_csv(f"data/intermediate/{dataset}/test.csv", header = None)[0].values


nodes = list(G.nodes())
labels = labels.loc[train].reindex(labels.index).fillna(0).astype('float64')


propagator=scaled_transition(G, nodes=nodes)
df,df_time=propagate(propagator, nodes, labels)
df = df.loc[test]
df.to_csv(f"data/processed/predictions/{dataset}/propagation_{name}_training_set_1.csv")
df_time.to_csv(f"data/processed/prediction_times/{dataset}/propagation_{name}_training_set_1.csv")
