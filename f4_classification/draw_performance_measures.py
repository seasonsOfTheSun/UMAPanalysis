


import os
os.chdir('//')

import pandas as pd
dataset = 'l1000'
df = pd.read_csv(f"performance_metrics/{dataset}/auc_for_prediction_methods.csv", index_col = 0)

element_dict = {}

width = 500
height = 100
path = "delete_this.svg"
df = df.sort_values("Propagation", ascending=False)

def performance_to_y_value(auc):
    return str(100 - 100*auc)

N = len(df.index)
def rank_to_x_value(i):
    return str(int(50+450*i/N))

def line(x1,x2,y1,y2):
    return {"@x1": str(x1), "@y1": str(y1),
     "@x2": str(x2),"@y2": str(y2),
     "@style":"stroke:rgb(0,0,0);stroke-width:2"}

element_dict['line'] = [line(30,30,0,100), line(30,470,100,100)]
element_dict['text'] = [{"@x":"15","@y":performance_to_y_value(1),"#text":"1.0", "@font-size":10, "@text-anchor":"end"},
                        {"@x":"15","@y":performance_to_y_value(0.5),"#text":"0.5", "@font-size":10, "@text-anchor":"end"}]

r = '2'
out = []
for i, moa in enumerate(df.index):
    v = df.loc[moa]
    x = rank_to_x_value(i)
    print(x)
    out.append({'@cx': x, '@cy': performance_to_y_value(v['Propagation']), '@r':r, '@fill':"black"})
    out.append({'@cx': x, '@cy': performance_to_y_value(v['RandomForest']), '@r':r, '@fill': '#00c2d7'})
    out.append({'@cx': x, '@cy': performance_to_y_value(v['NearestNeighbor']), '@r':r, '@fill': '#ae342b'})

element_dict['circle'] = out

out = []
for i,moa in enumerate(df.index):
    txt = " ".join([x.title() for x in moa.split("_")])
    out.append({"text":{"@x":rank_to_x_value(i),"@y":str(height + 5),"#text":txt, "@font-size":3, "@text-anchor":"end"},"@transform":"rotate(300 "+rank_to_x_value(i)+" "+str(height + 5)+")"})

out.append({'@transform':'translate(400 -50)',
            'circle':[{'@cx':0, '@cy':  0, '@r':r, '@fill':'#000000'},
                      {'@cx':0, '@cy': 15, '@r':r, '@fill':'#00c2d7'},
                      {'@cx':0, '@cy': 30, '@r':r, '@fill':'#ae342b'}],
            'text': [{'@x': 10, '@y':  3,  '#text':"Propagation",    "@font-size":10},
                     {'@x': 10, '@y': 18,  '#text':"Random Forest",  "@font-size":10},
                     {'@x': 10, '@y': 33,  '#text':"Nearest Neighbor", "@font-size":10}]})


x = "10"
y = "10"
out.append({"text": {"@x":x,"@y":y,"#text":"AUC","@font-size":20}, "@transform": "rotate(270 "+x+" "+y+")"})
element_dict['g'] = out

import xmltodict

fp = open(path, 'w')
element_dict['@viewBox'] = "0 0 "+str(width) + " " + str(height)
header  = {'svg': {'@xmlns':"http://www.w3.org/2000/svg", **element_dict}}
txt = xmltodict.unparse(header, pretty = True)
fp.write(txt)
fp.close()






import numpy as np
nearest_neighbor_times = pd.read_csv("performance_metrics/nearest_neighbor_prediction_times.csv", index_col = 0, header = None)
propagation_times = pd.read_csv("performance_metrics/propagation_prediction_times_nn_equal_4.csv", index_col = 0, header = None)
random_forest_times = pd.read_csv("performance_metrics/random_forest_prediction_times.csv", index_col = 0,header = None)

df['PropagationTimes'] = propagation_times[1]
# "black"
df['RandomForestTimes'] = random_forest_times[1]
#'#00c2d7'
df['NearestNeighborTimes'] = nearest_neighbor_times[1]
#'#ae342b'

import matplotlib.pyplot as plt
import seaborn as sns

fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8])
ax.scatter(df.Propagation, df.PropagationTimes, c = 'k')
ax.scatter(df.RandomForest, df.RandomForestTimes, c = '#00c2d7')
ax.scatter(df.NearestNeighbor, df.NearestNeighborTimes, c = '#ae342b')
ax.set_xlabel("AUROC")
ax.set_ylabel("time (s)")
ax.set_yscale('log')
ax.set_title("On leave-third-out classification task")
fig.savefig("figures/time_and_auroc_classification.png")
fig.savefig("figures/time_and_auroc_classification.svg")

fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8])
ax.scatter(df.RandomForest, df.Propagation, c = 'gray')
ax.plot([0.5, 1], [0.5, 1], color = 'k')
ax.set_xlabel("Random Forest AUROC")
ax.set_ylabel("Propagation AUROC")
ax.set_title("On leave-third-out classification task")
ax.set_aspect('equal')
ax.set_title("Performance")
fig.savefig("random_forest_propagation_auroc_comparison.png")
fig.savefig("random_forest_propagation_auroc_comparison.svg")

training = pd.read_csv("munged_data/l1000/moas_train.csv", index_col = 0)
fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8])
ax.scatter(df.Propagation / df.RandomForest, training.sum(axis = 0).loc[df.index].values, c = 'gray')

ax.set_xlabel("Propagation AUROC / Random Forest AUROC")
ax.set_ylabel("No. of training examples")
ax.set_title("On leave-third-out classification task")

ax.set_title("L1000 Performance")
fig.show()
fig.savefig("not_just_density.png")
fig.savefig("not_just_density.svg")


#save_svgdict(element_dict, height, width, path)