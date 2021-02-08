import pandas as pd
import

clusters = pandas.read_csv(f"data/intermediate/clusters/{method}_{dataset}.csv",index_col=0)
metadata = pandas.read_csv(f"data/intermediate/{dataset}/metadata.csv",index_col=0)
target = metadata.target

clusters = clusters[metadata.known==1]
target = target[metadata.known==1]

def cluster_dict(series):
    return {i:set(series[series == i].index) for i in series.unique()} 

def enrichment(set1, set2, n_total):
    return scipy.stats.hypergeom.pmf(len(set1&set2), n_total, len(set1), len(set2))

def cross_enrichment(dict1, dict2, N):
    out = {}
    for k1,v1 in dict1.items():
        out[k1]={}
        for k2,v2 in dict2.items():
            out[k1][k2] = enrichment(v1,v2,N)
    return pandas.DataFrame(out)

def bonferroni(enrichments):
    return enrichments * enrichments.size
