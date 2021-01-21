import pdb

import sklearn.ensemble as se
import pandas as pd
import time

os.chdir('/Users/jhancock/UMAP_analysis/')
dataset = 'l1000'

features = pd.read_csv(f"munged_data/{dataset}/features_train.csv", index_col = 0)
classes  = pd.read_csv(f"munged_data/{dataset}/moas_train.csv", index_col = 0)
metadata = pd.read_csv(f"munged_data/{dataset}/metadata.csv", index_col = 0)
metadata["known"] = metadata.known.astype('bool')

features = pd.read_csv(f"munged_data/{dataset}/features.csv", index_col = 0)
labels = pd.read_csv(f"munged_data/{dataset}/labels.csv", index_col = 0)


train = pd.read_csv(f"4.classification/train_and_test/{dataset}/train.csv", header = None)[0].values
test = pd.read_csv(f"4.classification/train_and_test/{dataset}/test.csv", header = None)[0].values

features_train = features.loc[train]
features_test = features.loc[test]
training_labels = labels.loc[train]
out = []
out_time = []
rg = se.RandomForestRegressor(n_estimators=10)
tick = 0
for x in labels.columns:
    print(x, tick)
    tick += 1
    start = time.time()
    rg.fit(features_train, training_labels[x])
    t = pd.Series(rg.predict(features_test))
    stop = time.time()
    out_time.append(stop-start)
    t.index = features_test.index
    t.name = x
    out.append(t)

pd.concat(out, axis = 1).to_csv(f"4.classification/predictions/{dataset}/random_forest_predictions.csv")
pd.Series(out_time, name="Time", index=classes.columns).to_csv("performance_metrics/random_forest_prediction_times.csv")
