
features = pd.read_csv(f"data/intermediate/{dataset}/features_train.csv", index_col = 0)
classes  = pd.read_csv(f"data/intermediate/{dataset}/moas_train.csv", index_col = 0)
metadata = pd.read_csv(f"data/intermediate/{dataset}/metadata.csv", index_col = 0)
metadata["known"] = metadata.known.astype('bool')


features = pd.read_csv(f"data/intermediate/{dataset}/features.csv", index_col = 0)
labels = pd.read_csv(f"data/intermediate/{dataset}/labels.csv", index_col = 0)


train = pd.read_csv(f"data/intermediate/{dataset}/train.csv", header = None)[0].values
test  = pd.read_csv(f"data/intermediate/{dataset}/test.csv", header = None)[0].values


features_train = features.loc[train]
features_test = features.loc[test]
training_labels = labels.loc[train]


out = []
rg = se.RandomForestRegressor(n_estimators=10)
for moa in labels.columns:
    rg.fit(features_train, training_labels[moa])
    t = pd.Series(rg.predict(features_test))

    t.index = features_test.index
    t.name = x
    out.append(t)

pd.concat(axis = 1).to_csv(f"data/processed/predictions/{dataset}_random_forest.csv")
