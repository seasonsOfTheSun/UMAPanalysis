
import pandas as pd
import numpy as np
import sys
import scipy.stats
import numpy
import re

filename = sys.argv[1]
m = re.match("data/processed/clusters/(?P<dataset>.*?)/(?P<name>.*?).csv", filename)
dataset = m.groupdict()['dataset']
name = m.groupdict()['name']


clusters = pd.read_csv(filename, index_col=0, header = None)
metadata = pd.read_csv(f"data/intermediate/{dataset}/metadata.csv"  ,index_col=0)
target = metadata.target


clusters = clusters[metadata.known==1]
target = target[metadata.known==1]


def cluster_dict(series):
    return {i:set(series[series == i].index) for i in series.unique()} 

def enrichment(set1, set2, n_total):
    pval = scipy.stats.hypergeom.cdf(len(set1&set2), n_total, len(set1), len(set2))
    return min(pval, 1-pval)
    
def cross_enrichment(dict1, dict2, N):
    out = {}
    for k1,v1 in dict1.items():
        out[k1]={}
        for k2,v2 in dict2.items():
            out[k1][k2] = enrichment(v1,v2,N)
    return pd.DataFrame(out)

def odds_ratio(condition, outcome, N):
    try:
        num = len(condition & outcome)/len(condition - outcome)
        den = len(outcome - condition)/(N-len(condition - outcome))
        return num/den
    except ZeroDivisionError:
        return np.nan

def cross_odds_ratio(dict1, dict2, N):
    out = {}
    for k1,v1 in dict1.items():
        out[k1]={}
        for k2,v2 in dict2.items():
            out[k1][k2] = odds_ratio(v1,v2,N)
    return pd.DataFrame(out)

def bonferroni(enrichments):
    return enrichments * enrichments.size


enrichment_list = []
odds = []
for clust in clusters.columns:
    c = cluster_dict(clusters[clust])
    d = cluster_dict(target)
    n = sum(metadata.known==1)
    enrichment_list.append(cross_enrichment(c,d,n))
    odds.append(cross_odds_ratio(c,d,n))
    
df = pd.concat(enrichment_list, axis = 1)
df.to_csv(f"data/processed/cluster_enrichments/{dataset}/enrichments_{name}.csv")

df = pd.concat(odds, axis = 1)
df.to_csv(f"data/processed/cluster_enrichments/{dataset}/odds_ratios_{name}.csv")
