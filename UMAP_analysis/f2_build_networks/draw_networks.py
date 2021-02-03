
import matplotlib.pyplot as plt
import networkx as nx

G = nx.read_gml("networks/l1000/similarity.gml")

fig= plt.figure(figsize = [10,10])
ax = fig.add_axes([0,0,1,1])
ax.set_xticks([])
ax.set_yticks([])
collec = PolyCollection([[layout[i], layout[j], layout[k]] for i,j,k in simplices], facecolors = ["#00000044" for i in range(len(simplices))])
ax.add_collection(collec)
nx.draw_networkx(G.to_undirected(),
                 ax = ax,
                 pos = layout,
                 labels = {},
                 node_color = [d[spp] for i in alpha],
                 node_size = list(300*alpha))

fig.save("mpthx.svg")
