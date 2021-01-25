
import pandas as pd
import networkx as nx
import time

os.chdir('/Users/jhancock/UMAP_analysis/')

classes = pd.read_csv(f"munged_data/{dataset}/labels.csv", index_col = 0)
metadata = pd.read_csv(f"munged_data/{dataset}/metadata.csv", index_col = 0)
metadata["known"] = metadata.known.astype('bool')

classes_test = pd.read_csv(f"munged_data/{dataset}/labels.csv", index_col = 0)
similarity = nx.read_gml(f"networks/{dataset}/similarity.gml")

out = []
out_time = []
tick = 0
for x in classes.columns:
    print(x, tick)
    tick += 1
    temp = []
    start = time.time()
    for node in classes_test.index:
        neighbors = similarity[node]
        neighbor_props = metadata.loc[list(neighbors.keys())]
        known_neighbors = list(neighbor_props[neighbor_props.known == True].index)
        if len(known_neighbors) > 0:
            nearest_known_neighbor = max(known_neighbors, key=lambda x: similarity[node][x]['weight'])
            prediction = classes.loc[nearest_known_neighbor, x]
        else:
            prediction = 0
        temp.append(prediction)
    stop = time.time()
    t = pd.Series(temp)
    t.index = classes_test.index
    t.name = x
    out.append(t)
    out_time.append(stop-start)


pd.concat(out, axis = 1).to_csv(f"4.classification/predictions/{dataset}/nearest_neighbor_predictions.csv")
pd.Series(out_time, name="Time", index=classes.columns).to_csv(f"4.classification/performance_metrics/nearest_neighbor_prediction_times.csv")