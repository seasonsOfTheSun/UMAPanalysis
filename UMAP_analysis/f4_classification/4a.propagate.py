
import networkx as nx
import scipy.sparse
import numpy as np
import pandas as pd
import time

# features = pd.read_csv("features_train.csv", index_col = 0)


def make_propagator(G, restart=0.1):
    nodes = list(G.nodes())

    A=nx.adjacency_matrix(G)
    temp=G.out_degree(weight='weight')
    scaled = scipy.sparse.diags([1/temp[i] for i in nodes]) * A


    scaled = (1-restart) * scaled
    propagator = scipy.sparse.eye(len(nodes)) + scaled + scaled**2 + scaled**3
    return propagator,nodes


def propagate(propagator, nodes, labels):
    out = []
    out_time = []
    for i,x in enumerate(labels.columns):
        print(i,x)
        start = time.time()
        v = scipy.sparse.csc_matrix(labels.loc[list(nodes), x].values.reshape(-1, 1))
        temp = pd.Series((propagator * v).toarray().ravel())
        temp.name = x
        out.append(temp)
        stop = time.time()
        out_time.append(stop-start)

    df = pd.concat(out, axis=1)
    df.index = nodes
    df_time = pd.Series(out_time, name="Time", index=labels.columns)

    return df, df_time


import os
os.chdir('//')


dataset = 'lish-moa'
G=nx.read_gml(f"networks/{dataset}/similarity.gml")
labels=pd.read_csv(f"data/intermediate/{dataset}/labels.csv", index_col=0)
metadata = pd.read_csv(f"data/intermediate/{dataset}/metadata.csv", index_col=0)
features = pd.read_csv(f"data/intermediate/{dataset}/features.csv", index_col=0)

train = pd.read_csv(f"4.classification/train_and_test/{dataset}/train.csv", header = None)[0].values
test = pd.read_csv(f"4.classification/train_and_test/{dataset}/test.csv", header = None)[0].values

testing = pd.Series({i:(i in test) for i in labels.index})
labels = labels.mask(testing, other=0)

propagator,nodes=make_propagator(G)
df,df_time=propagate(propagator, nodes, moas)
df.to_csv(f"predictions/{dataset}/predicted_by_propagation.csv")
df_time.to_csv(f"performance_metrics/{dataset}/predicted_by_propagation.csv")
