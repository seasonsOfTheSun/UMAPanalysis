

import os
os.chdir('//')

import pandas as pd
import networkx as nx
import numpy as np
import scipy.sparse
import xmltodict

dataset = 'l1000'
labels = pd.read_csv(f"munged_data/{dataset}/labels.csv", index_col = 0)

G = nx.read_gml(f"//networks/{dataset}/similarity.gml")
labels = labels.loc[sorted(G.nodes())]
Gc = nx.read_gml(f"//networks/{dataset}/coarsened_similarity.gml")
coarsen = scipy.sparse.load_npz(f"//networks/{dataset}/similarity_coarsening_map.npz")


coarsened_labels = pd.DataFrame(coarsen * labels)
coarsened_labels.columns = labels.columns
coarsened_labels.index = coarsened_labels.index.astype('str')
#nx.draw_networkx(Gc, with_labels = False, node_size = 5*coarsen.sum(axis = 1).A.flatten())


import xmltodict
import numpy as np

def save_svg(path, element_list, width = 100, height = 100):
    element_dict = {'g':element_list}
    fp = open(path, 'w')
    element_dict['@viewBox'] = "0 0 "+str(width) + " " + str(height)
    header = {'svg': {'@xmlns': "http://www.w3.org/2000/svg", **element_dict}}
    txt = xmltodict.unparse(header, pretty=True)
    fp.write(txt)
    fp.close()


def sector(centre, radius, start_angle, end_angle, fill="black", stroke="none"):
    centre_x,centre_y = centre

    start_x = centre_x + radius * np.cos(np.deg2rad(start_angle))
    start_y = centre_y + radius * np.sin(np.deg2rad(-start_angle))

    end_x = centre_x + radius * np.cos(np.deg2rad(end_angle))
    end_y = centre_y + radius * np.sin(np.deg2rad(-end_angle))

    path_string = "M {0} {1} A {2} {2} 0 0 0 {3} {4}".format(start_x, start_y, radius, end_x, end_y)
    path_string += "L {0} {1} Z".format(centre_x, centre_y)
    return {"path": {"@d": path_string, "@stroke": stroke, "@fill": fill}}


def sectored_circle(centre, radius, angle_list, color_list, stroke="none"):
    angle_list_shift = angle_list[1:] + [angle_list[0]]

    out = []
    for start_angle, end_angle, color in zip(angle_list, angle_list_shift, color_list):
        temp = sector(centre, radius, start_angle, end_angle, fill=color, stroke=stroke)
        out.append(temp)

    return {"g": out}

def pie_chart(series_, to_color, color_key):
    total = series_.sum()
    scaled_series = 360 * series_ / total
    scaled_series = scaled_series.loc[to_color]
    angle_positions = scaled_series.cumsum()
    return total, angle_positions, [color_key[i] for i in to_color]
def write(x,y,text,size):return {'text': {'@x':x,'@y':y,'#text':text,"@font-size":size}}
def circle(x,y,r,c,stroke='none', width = 1): return {'circle':{'@cx':x,'@cy':y,'@r':r,'@fill':c,'@stroke':stroke,'@style':"stroke-width:{0}".format(width)}}
def line(x1,x2,y1,y2,c="#000000",width=1):
    return {'line':{"@x1": x1, "@y1": y1, "@x2": x2,"@y2": y2,"@style":"stroke:"+c+";stroke-width:"+str(width)}}

Gc_sparser = Gc.copy()
for edge in Gc.edges():
    if Gc.edges()[edge]['weight'] < 2:
        Gc_sparser.remove_edge(edge[0], edge[1])


pos = nx.kamada_kawai_layout(Gc_sparser, weight=None)

def axis_map(arr):
    return (25+50*arr[0],25+50*arr[1])

preferred = coarsened_labels.sum().sort_values()[-5:].index


colormap = {i: "#"+hex(np.random.randint(16**6)).ljust(8, '0')[2:] for i in preferred}

out = []
for edge in Gc.edges():
    x1, y1 = axis_map(pos[edge[0]])
    x2, y2 = axis_map(pos[edge[1]])
    out.append(line(x1, x2, y1, y2, c="#00000020", width=0.1))

for node in coarsened_labels.sum(axis = 1).sort_values().index:


    radius_unscaled, angle_list,color_list = pie_chart(coarsened_labels.loc[node], preferred, colormap)
    radius = 0.5 * np.sqrt(radius_unscaled)
    centre = axis_map(pos[node])

    d = sectored_circle(centre, radius, angle_list, color_list, stroke= "none")
    c1 = circle(centre[0],centre[1],radius,'white',stroke='none', width = 0.3)
    c2 = circle(centre[0], centre[1], radius, 'none', stroke='black', width=0.1)
    out.append(c1)
    out.append(d)
    out.append(c2)

save_svg("quart.svg", out, width = 100, height = 100)

out = []
y = 10
for i,v in colormap.items():
    out.append(circle(20, y, 4, v))
    out.append(write(25, y+3, " ".join(i.split("_")).title(), 10))
    y += 10

save_svg("legend.svg", out, width = 200, height = 100)


max_v = max(coarsened_labels.sum(axis = 1))
min_v= min(coarsened_labels.sum(axis = 1))

out = []
y = 10
for radius_unscaled in np.linspace(min_v, max_v, 5):
    radius = 0.5 * np.sqrt(radius_unscaled)
    out.append(circle(20, y, radius,'none', stroke='black', width=0.1))
    out.append(write(25, y+3, str(round(radius_unscaled, 2)), 5))
    y += 10

save_svg("size_legend.svg", out, width = 200, height = 100)