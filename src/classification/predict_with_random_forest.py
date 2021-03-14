import pandas as pd
import sys
import sklearn.ensemble as se
import time
import numpy as np

dataset = sys.argv[1]
moa_file = sys.argv[2]
features = pd.read_csv(f"data/intermediate/{dataset}/features.csv", index_col = 0)

true_labels = pd.read_csv(f"data/intermediate/{dataset}/labels.csv", index_col = 0)
train = pd.read_csv(f"data/intermediate/{dataset}/train.csv", header = None)[0].values
test = pd.read_csv(f"data/intermediate/{dataset}/test.csv", header = None)[0].values


not_masked = [i in train for i in true_labels.index]
not_masked = pd.Series(not_masked, index = true_labels.index)
labels = true_labels.mask(~not_masked).fillna(0)


fp = open(moa_file)
moa = fp.readline()

print(time.ctime())
rg = se.RandomForestRegressor(n_estimators=10)
rg.fit(features.loc[train], true_labels.loc[train, moa])
out = pd.Series(rg.predict(features.loc[test]))
print(time.ctime())
out.index = test
out.name = moa


iteration = str(np.random.randint(10**16-1)).rjust(16, '0')
moa_file_id = moa_file.split("/")[-1]
out.to_csv(f"data/processed/predictions/{dataset}/random_forest_{moa_file_id}-{iteration}.csv")
