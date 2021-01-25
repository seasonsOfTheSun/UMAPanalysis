
import pandas as pd
import networkx as nx
import numpy as np

import os
os.chdir('//')

dataset = "cytodata"

for dataset in ['lish-moa', 'cytodata', 'GDSC']:
    G = nx.read_gml(f"networks/{dataset}/similarity.gml")

    node = list(G.nodes())[0]
    neighbors = np.random.choice(G.nodes(), int(len(G.nodes)/3))
    G_sub = G.subgraph(neighbors)
    ecentre = nx.eigenvector_centrality_numpy(G)
    ecentre = pd.Series(ecentre)
    ecentre.to_csv(f"5.centrality/eigencentrality/{dataset}/eigencentrality.csv")