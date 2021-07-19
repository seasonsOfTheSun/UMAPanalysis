import synthetic
import methods
import numpy as np

cell_line_dataset = methods.load_from_file("../../data/intermediate/cell_line/features.csv",
                                  "../../data/intermediate/cell_line/labels.csv",
                                  "../../networks/cell_line/metric_manhattan_nneighbors_10.gml",
                                  "../../networks/cell_line/evaluation_time_metric_manhattan_nneighbors_10.gml",
                                  column = "MeSH"
                                 )

metadata_df = methods.clustering_method_dataframe(methods.clustering_methods)
metadata_df.index.name = "clustering methods"
score_df, time_df, n_df = methods.evaluate(methods.clustering_methods, dcell_line_dataset)

import os
dataset_series.save("src/clustering/cell_line/")
os.makedirs("src/clustering/cell_line", exist_ok = True)
score_df.to_csv("src/clustering/cell_line/score_df.csv")
time_df.to_csv("src/clustering/cell_line/time_df.csv")
n_df.to_csv("src/clustering/cell_line/n_df.csv")
metadata_df.to_csv("src/clustering/cell_line/metadata_df.csv")