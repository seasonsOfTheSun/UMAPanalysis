import pandas
import sys
import scipy.stats
import numpy
import re

filename = sys.argv[1]
m = re.match("data/processed/clusters/(?P<dataset>.*?)/(?P<name>.*?).csv", filename)
dataset = m.groupdict()['dataset']
name = m.groupdict()['name']


clusters = pandas.read_csv(filename, index_col=0, header = None)
metadata = pandas.read_csv(f"data/intermediate/{dataset}/metadata.csv"  ,index_col=0)
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
    return pandas.DataFrame(out)

def odds_ratio(set1, set2, N):
    num = len(set1&set2)*len(set2)
    den = len(set2-set1)*(N-len(set1))
    return num/den

def cross_odds_ratio(dict1, dict2, N):
    out = {}
    for k1,v1 in dict1.items():
        out[k1]={}
        for k2,v2 in dict2.items():
            out[k1][k2] = odds_ratio(v1,v2,N)
    return pandas.DataFrame(out)

def bonferroni(enrichments):
    return enrichments * enrichments.size


enrichment = []
odds = []
for clust in clusters.columns[2]:
    c = cluster_dict(clusters[clust])
    d = cluster_dict(target)
    n = sum(metadata.known==1)
    enrichment.append(bonferroni(cross_enrichment(c,d,n)))
    odds.append(cross_odds_ratio(c,d,n))
    
df = pd.concat(enrichment, axis = 1)
df.to_csv(f"data/processed/cluster_enrichments/{dataset}/enrichments_{name}.csv")

df = pd.concat(odds, axis = 1)
df.to_csv(f"data/processed/cluster_enrichments/{dataset}/odds_ratios_{name}.csv")
