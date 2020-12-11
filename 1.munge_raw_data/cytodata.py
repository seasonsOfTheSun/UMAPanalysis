
import pandas as pd
import numpy as np

import os
os.chdir('//')

#df = pd.read_csv("lish-moa/sample_submission.csv", index_col = 0)
X_unclassified = pd.read_csv("cytodata/validation_data.csv");
X_classified = pd.read_csv("cytodata/training_data.csv");

X_classified['known'] = 1
X_unclassified['known'] = 0

X_unclassified = X_unclassified.set_index("cell_code")
X_classified = X_classified.set_index("cell_code")
X = pd.concat([X_classified, X_unclassified]);

targets = list(X.target.unique())
targets.remove(np.nan)
y = pd.concat([X.target == target for target in targets], axis = 1)
y.fillna(0.0)
y.columns = targets
y.astype('int')

metadata_columns =  [
                     'cell_id',
                     'dist.10.nn',
                     'dist.20.nn',
                     'dist.30.nn',
                     'field',
                     'nuclear.displacement',
                     'plate',
                     'replicate',
                     'target',
                     'well',
                     'well_code',
                     'known'
                    ]
metadata = X[metadata_columns]
features = X[[i for i in X.columns if i not in metadata_columns]]

features_unscaled = features.copy()
features = features_unscaled / features_unscaled.std(axis = 0)

features.to_csv("munged_data/cytodata/features.csv")
features_unscaled.to_csv("munged_data/cytodata/features_unscaled.csv")
metadata.to_csv("munged_data/cytodata/metadata.csv")
y.to_csv("munged_data/cytodata/labels.csv")
