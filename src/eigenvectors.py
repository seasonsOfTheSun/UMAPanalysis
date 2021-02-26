import re
import networkx as nx

filename = sys.argv[1]
expr = "networks/(?P<dataset>.*?)/metric_(?P<metric>.*?)_nneighors_(?P<n>\d*?).gml")
m = re.match(exp, filename)
dataset = m.groupdict()['dataset']
metric = m.groupdict()['metric']
n = m.groupdict()['n']


G = nx.read_gml(filename)
transition = scaled_transition(G, nodes)
evecs = eigenvalues(transition, nodes)
evecs.to_csv( f"networks/{dataset}/eigenvectors/transition_metric_{metric}_nneighors_{n}.csv")


laplacian = scaled_transition(G, nodes)
evecs = eigenvalues(laplacian, nodes)
evecs.to_csv( f"networks/{dataset}/eigenvectors/laplacian_metric_{metric}_nneighors_{n}.csv")
