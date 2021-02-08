import pandas as pd
import sys
import sklearn.ensemble as se

dataset = sys.argv[1]

features = pd.read_csv(f"data/intermediate/{dataset}/features.csv", index_col = 0)
labels = pd.read_csv(f"data/intermediate/{dataset}/labels.csv", index_col = 0)


train = pd.read_csv(f"data/intermediate/{dataset}/train.csv", header = None)[0].values
test  = pd.read_csv(f"data/intermediate/{dataset}/test.csv", header = None)[0].values


features_train = features.loc[train]
features_test = features.loc[test]
training_labels = labels.loc[train]

df = []
for moa in training_labels.columns:
    rg = se.RandomForestRegressor(n_estimators=10)
    rg.fit(features_train, training_labels[moa])
    out = pd.Series(rg.predict(features_test))
    out.index = features_test.index
    out.name = moa
    df.append(out)


df = pd.concat(df, axis=1)
df.to_csv(f"data/processed/predictions/{dataset}/random_forest_predictions.csv")
