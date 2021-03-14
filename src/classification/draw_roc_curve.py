import re
import sys
import pandas as pd
from UMAP_analysis.stats_utils.receiver_operating_characteristic import *

filename = sys.argv[1]
m = re.match("data/processed/predictions/(?P<dataset>.*?)/(?P<prediction>.*?).csv", filename)
dataset = m.groupdict()['dataset']
prediction_str = m.groupdict()['prediction']

test  = pd.read_csv(f"data/intermediate/{dataset}/test.csv", header = None)[0].values
truth_full = pd.read_csv(f"data/intermediate/{dataset}/labels.csv", index_col=0)
prediction = pd.read_csv(f"data/processed/predictions/{dataset}/{prediction_str}.csv", index_col=0)
truth = truth_full.loc[test]

out = []
for moa in prediction.columns:
    roc = roc_curve(truth[moa], prediction[moa])
    roc.to_csv(f"data/processed/rocs/{dataset}/{prediction_str}_label_{moa}.csv")
    auc = auc_from_roc(roc.FPR, roc.TPR)
    out.append(auc)
    print(f"data/processed/rocs/{dataset}/{prediction_str}_label_{moa}.csv")

    
aucs = pd.Series(out, index = prediction.columns)
aucs.to_csv(f"data/processed/aucs/{dataset}/{prediction_str}.csv")
