import synthetic
import methods
import numpy as np

morphological_dataset = methods.load_from_file("data/intermediate/morphological/features.csv",
                                  "data/intermediate/morphological/labels.csv",
                                  "networks/morphological/metric_euclidean_nneighbors_10.gml",
                                  "networks/morphological/evaluation_time_metric_euclidean_nneighbors_10.gml",
                                  column = "MeSH"
                                 )
                                 
metadata_df = methods.clustering_method_dataframe(methods.clustering_methods)
metadata_df.index.name = "clustering methods"
score_df, time_df, n_df = methods.evaluate(methods.clustering_methods, morphological_dataset)

import os
dataset_series.save("src/clustering/morphology/")
os.makedirs("src/clustering/morphology", exist_ok = True)

score_df.to_csv("src/clustering/morphology/score_df.csv")
time_df.to_csv("src/clustering/morphology/time_df.csv")
n_df.to_csv("src/clustering/morphology/n_df.csv")
metadata_df.to_csv("src/clustering/morphology/metadata_df.csv")
