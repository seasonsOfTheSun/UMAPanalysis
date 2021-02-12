import UMAP_analysis.stats_utils.enrichment

import pandas as pd
import sys
import re
import pandas
import os

dataset = sys.argv[1]

methods = set()
for filename in os.listdir(f"../data/processed/clusters/{dataset}/"):
    m = re.match("(?P<method>.*?)_noise_percent_(?P<noise_percent>\d*?)\.csv", filename)
    method = m.groupdict()['method']
    methods |= {method}



metadata = pandas.read_csv(f"../data/intermediate/{dataset}/metadata.csv",index_col=0)
target = metadata.target
target = target[metadata.known==1]
N = len(metadata[metadata.known==1].index)

out = {method:[] for method in methods}
for filename in os.listdir(f"../data/processed/clusters/{dataset}/"):
    m = re.match("(?P<method>.*?)_noise_percent_(?P<noise_percent>\d*?)\.csv", filename)
    method = m.groupdict()['method']
    noise_level = int(m.groupdict()['noise_percent'])/100
    
    clusters = pandas.read_csv(f"../data/processed/clusters/{dataset}/"+filename,index_col=0, header = None)[1]
    clusters = clusters[metadata.known==1]

    pvalues_df = UMAP_analysis.stats_utils.enrichment.pairwise_enrichment_adjusted(clusters, target, N)
    temp = pvalues_df[0]
    temp.name = noise_level
    out[method].append(temp.copy())
    

for method in methods:
    pd.DataFrame(out[method]).to_csv(f"../data/processed/cluster_enrichment/{dataset}/"+method+".csv")
