
import pandas as pd
import numpy as np


X1 = pd.read_excel("data/raw/GDSC/GDSC1_fitted_dose_response_25Feb20.xlsx")
X2 = pd.read_excel("data/raw/GDSC/GDSC2_fitted_dose_response_25Feb20.xlsx")

X1_t = X1[["CELL_LINE_NAME", "DRUG_NAME", "LN_IC50"]]
X1_t = X1_t.groupby(["CELL_LINE_NAME", "DRUG_NAME"]).sum()
X1_t = X1_t.reset_index()
X1_t = X1_t.pivot(index= "DRUG_NAME", columns = "CELL_LINE_NAME", values = "LN_IC50")

X2_t = X2[["CELL_LINE_NAME", "DRUG_NAME", "LN_IC50"]]
X2_t = X2_t.groupby(["CELL_LINE_NAME", "DRUG_NAME"]).sum()
X2_t = X2_t.reset_index()
X2_t = X2_t.pivot(index= "DRUG_NAME", columns = "CELL_LINE_NAME", values = "LN_IC50")

#y_unrefined = pd.read_csv("NCI-ALMANAC/ComboDrugGrowth_Nov2017.csv", index_col = 0)

X = pd.concat([X1_t, X2_t])
X = X.fillna(X.median(axis =1))

X['known'] = 1

metadata_columns = ['known']
metadata = X[metadata_columns]
features = X[[i for i in X.columns if i not in metadata_columns]]

features_unscaled = features.copy()
features = features_unscaled / features_unscaled.std(axis = 0)

features.to_csv("data/intermediate/GDSC/features.csv")
features_unscaled.to_csv("data/intermediate/GDSC/features_unscaled.csv")
metadata.to_csv("data/intermediate/GDSC/metadata.csv")
#y.to_csv("data/intermediate/GDSC/labels.csv")
