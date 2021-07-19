import synthetic
import methods
import numpy as np

transcriptional_dataset = methods.load_from_file("../../data/intermediate/transcriptional/features.csv",
                                  "../../data/intermediate/transcriptional/labels.csv",
                                  "../../networks/transcriptional/metric_manhattan_nneighbors_10.gml",
                                  "../../networks/transcriptional/evaluation_time_metric_manhattan_nneighbors_10.gml",
                                  column = "MeSH"
                                 )


metadata_df = methods.clustering_method_dataframe(methods.clustering_methods)
metadata_df.index.name = "clustering methods"
score_df, time_df, n_df = methods.evaluate(methods.clustering_methods, transcriptional_dataset)

import os
dataset_series.save("src/clustering/transcription/")
os.makedirs("src/clustering/transcription", exist_ok = True)

score_df.to_csv("src/clustering/transcription/score_df.csv")
time_df.to_csv("src/clustering/transcription/time_df.csv")
n_df.to_csv("src/clustering/transcription/n_df.csv")
metadata_df.to_csv("src/clustering/transcription/metadata_df.csv")