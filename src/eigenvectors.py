import re
import networkx as nx
form UMAP_analsis.core import scaled_transition

filename = sys.argv[1]
expr = "networks/(?P<dataset>.*?)/(?P<name>.*?).gml"
m = re.match(exp, filename)
dataset = m.groupdict()['dataset']
name = m.groupdict()['name']

G = nx.read_gml(filename)
transition = scaled_transition(G, nodes)
evecs = eigenvalues(transition, nodes)
evecs.to_csv(f"networks/{dataset}/eigenvectors/transition_{name}.csv")


laplacian = scaled_transition(G, nodes)
evecs = eigenvalues(laplacian, nodes)
evecs.to_csv(f"networks/{dataset}/eigenvectors/laplacian_{name}.csv")
