import umap
import numpy as np
import pandas as pd
import networkx as nx

def umap_network(df, nn):
    rndstate = np.random.RandomState()
    knn_net = umap.umap_.fuzzy_simplicial_set(df.values, nn, rndstate, 'manhattan')
    G = nx.from_numpy_array(knn_net[0].toarray(), create_using = nx.DiGraph)
    num_to_id = dict(enumerate(df.index))
    return nx.relabel_nodes(G, num_to_id.get)

def scaled_laplacian(G):
    components = nx.weakly_connected_component_subgraphs(G)
    G = max(components, key = lambda H: H.order())
    nodes = G.nodes()

    A = nx.adjacency_matrix(G)
    temp = G.out_degree(weight = 'weight')
    scaled = scipy.sparse.diags([1/temp[i] for i in nodes]) * A
    return scaled

def eigenvalues(scaled):
    eval_, evec = scipy.sparse.linalg.eigs(scaled, k = 1000)

    eval_ = np.real(eval_)
    evec = np.real(evec)

    evec = pd.DataFrame(evec)
    evec.index = nodes
    evec.loc["lambda",:] = eval_
    evec[np.argsort(eval_)[::-1]]
    return evec
