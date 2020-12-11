
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pdb
def true_positive_rate(truth, predicted):
    return sum(truth & predicted) / sum(truth)

def false_positive_rate(truth, predicted):
    #pdb.set_trace()
    return sum(~truth & predicted) / sum(~truth)

def roc_curve(truth, predicted_values, steps = 500):

    x = []
    y = []

    max_p = max(predicted_values)
    min_p = min(predicted_values)

    predicted = predicted_values >= min_p
    x.append(false_positive_rate(truth, predicted))
    y.append(true_positive_rate(truth, predicted))

    for threshold in np.linspace(min_p, max_p, steps):
        predicted = predicted_values > threshold
        x.append(false_positive_rate(truth, predicted))
        y.append(true_positive_rate(truth, predicted))

    df = pd.DataFrame([x, y]).T
    df.columns = ["FPR", "TPR"]
    df["threshold"] = [min_p]+list(np.linspace(min_p, max_p, steps))
    return df

import os
os.chdir('//')
dataset = "l1000"
truth = pd.read_csv(f"munged_data/{dataset}/labels.csv", index_col=0)

propagation_prediction = pd.read_csv(f"predictions/{dataset}/predicted_by_propagation.csv", index_col=0)
random_forest_prediction = pd.read_csv(f"predictions/{dataset}/random_forest_predictions.csv", index_col=0)
nearest_neighbor_prediction = pd.read_csv(f"predictions/{dataset}/nearest_neighbor_predictions.csv", index_col=0)

#this is an embaressment
nearest_neighbor_prediction = nearest_neighbor_prediction[~nearest_neighbor_prediction.index.duplicated(keep = 'first')]
random_forest_prediction = random_forest_prediction[~random_forest_prediction.index.duplicated(keep = 'first')]
truth = truth.loc[random_forest_prediction.index]


def auc_from_roc(x_curve,y_curve):

    auc = 0
    for i,x,y in zip(range(len(x_curve)), x_curve, y_curve):

        if i == 0:
            x_prev = x
            y_prev = y
            continue

        auc += (x_prev - x) * (y+y_prev)/2
        x_prev = x
        y_prev = y

    return auc


moas = set(propagation_prediction.columns) & set(random_forest_prediction.columns)

# propagation   #000000
# random forest #00c2d7
# other method  #ae342b

auc_propagation = []
auc_random_forest = []
auc_nearest_neighbor = []

#fig, axes = plt.subplots(nrows=7, ncols=int(len(moas)/7+1), figsize=[20,30])
i = 0
steps = 100
for moa in moas:
    print(moa, '  ', i, '/', len(moas))
    i += 1

    if ~(truth[moa] > 0).any():
        continue

    df_random_forest = roc_curve(truth[moa] > 0, random_forest_prediction.loc[truth.index, moa], steps = steps)
    auc_random_forest.append(auc_from_roc(df_random_forest.FPR, df_random_forest.TPR))
    df_random_forest.columns = "random_forest_"+df_random_forest.columns

    df_nearest_neighbor = roc_curve(truth[moa] > 0, nearest_neighbor_prediction.loc[truth.index, moa], steps = steps)
    auc_nearest_neighbor.append(auc_from_roc(df_nearest_neighbor.FPR, df_nearest_neighbor.TPR))
    df_nearest_neighbor.columns = "nearest_neighbor_" + df_nearest_neighbor.columns

    df_propagation = roc_curve(truth[moa] > 0, propagation_prediction.loc[truth.index, moa], steps = steps)
    auc_propagation.append(auc_from_roc(df_propagation.FPR, df_propagation.TPR))
    df_propagation.columns = "propagation_"+df_propagation.columns

    df = pd.concat([df_propagation, df_nearest_neighbor, df_random_forest], axis=1, sort = False)
    df.to_csv("performance_metrics/moa_csvs/"+moa+"_roc_curves.csv")



df = pd.DataFrame([auc_propagation, auc_random_forest, auc_nearest_neighbor]).T
df.columns = ["Propagation", "RandomForest", "NearestNeighbor"]
df.index = [moa for moa in moas if (truth[moa] > 0).any()]
df.to_csv(f"performance_metrics/{dataset}/auc_for_prediction_methods.csv")
