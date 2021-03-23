import networkx as nx
import pandas as pd
import pygsp
import graph_coarsening as gc
import scipy.sparse
import sys

condensed_moas.columns = moas.columns
filename = sys.argv[1]
G_nx = nx.read_gml(filename)


def coarsen(G_nx, r=0.99):
    G_nx = max(nx.weakly_connected_component_subgraphs(G_nx), key = lambda x:len(x.nodes()))
    nodes = sorted(list(G_nx.nodes()))
    G = pygsp.graphs.Graph(nx.adjacency_matrix(G_nx, nodelist=nodes))

    C,Gc,*_ = gc.coarsen(G, r = r)
    Gc = nx.from_scipy_sparse_matrix(Gc.W)
    pos = nx.spring_layout(Gc)
    Gc_nodes = list(sorted(Gc.nodes()))
    return Gc


def condense(C, nodes):
    moas = moas.reindex(nodes)
    moa_matrix = scipy.sparse.csr_matrix(moas.values)
    condensed_moas = pd.DataFrame((C * moa_matrix).toarray())
    return condensed_moas


nx.write_gml(Gc,"networks/coarsened/")
scipy.sparse.save_npz("networks/coarsening_matrix/", C)
