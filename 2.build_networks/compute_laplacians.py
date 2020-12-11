import networkx as nx
import scipy.sparse
import numpy as np
import pandas as pd

os.chdir('/Users/jhancock/UMAP_analysis/')

G = nx.read_gml("networks_nn_equal_4/similarity.gml")
components = nx.weakly_connected_component_subgraphs(G)
G = max(components, key = lambda H: H.order())
nodes = G.nodes()

A = nx.adjacency_matrix(G)
temp = G.out_degree(weight = 'weight')
scaled = scipy.sparse.diags([1/temp[i] for i in nodes]) * A

eval_, evec = scipy.sparse.linalg.eigs(scaled, k = 1000)

eval_ = np.real(eval_)
evec = np.real(evec)

evec = pd.DataFrame(evec)
evec.index = nodes
evec.loc["lambda",:] = eval_
evec[np.argsort(eval_)[::-1]]

evec.to_csv("networks_nn_equal_4/normalized_laplacian_eigenvectors.csv")
