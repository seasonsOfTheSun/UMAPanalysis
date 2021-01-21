
import umap
import numba
import numpy as np
import pandas as pd
import networkx as nx

def umap_network(df, nn):
    rndstate = np.random.RandomState()
    knn_net = umap.umap_.fuzzy_simplicial_set(df.values, nn, rndstate, 'manhattan')
    G = nx.from_numpy_array(knn_net[0].toarray(), create_using = nx.DiGraph)
    num_to_id = dict(enumerate(df.index))
    return nx.relabel_nodes(G, num_to_id.get)

if __name__ == '__main__':
    n = 14
    #for dataset in ['GDSC','cytodata','lish-moa']:
    for dataset in ['toy']:
        print()
        df = pd.read_csv(f"munged_data/{dataset}/features.csv", index_col=0)
        G = umap_network(df, n)
        nx.write_gml(G, f"networks/{dataset}/similarity.gml")

