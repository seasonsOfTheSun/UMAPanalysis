import re
import networkx as nx
from UMAP_analysis.core import scaled_transition

filename = sys.argv[1]
expr = "networks/(?P<dataset>.*?)/(?P<name>.*?).gml"
m = re.match(exp, filename)
dataset = m.groupdict()['dataset']
name = m.groupdict()['name']

G = nx.read_gml(filename)
nodes = list(G.nodes())
transition = scaled_transition(G, nodes)
evecs = eigenvalues(transition, nodes)
evecs.to_csv(f"networks/{dataset}/eigenvectors/transition_{name}.csv")


laplacian = scaled_laplacian_opposite(G, nodes)
evecs = eigenvalues(laplacian, nodes)
evecs.to_csv(f"networks/{dataset}/eigenvectors/laplacian_{name}.csv")
