import networkx as nx


def kamada_kawaii(G):
    j = None
    G_h = G.copy()
    for s in nx.weakly_connected_components(G):
        i = s.pop()
        if j == None:
            pass
        else:
            G_h.add_edge(i, j)
        j = i
    return nx.layout.kamada_kawai_layout(G_h, weight=None)


def draw(G):
    fig= plt.figure(figsize = [10,10])
    ax = fig.add_axes([0,0,1,1])
    ax.set_xticks([])
    ax.set_yticks([])
    simplices = [i for i in nx.clique.find_cliques(G.to_undirected()) if len(i) == 3]
    collec = PolyCollection([[layout[i],layout[j],layout[k]] for i,j,k in simplices], facecolors = ["#00000044" for i in range(len(simplices))])
    ax.add_collection(collec)
    nx.draw_networkx(G,
                     ax = ax,
                     pos = layout,
                     labels = {},
                     node_color = df.Species.map(d.get))
