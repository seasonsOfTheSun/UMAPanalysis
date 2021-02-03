import pandas
import scipy.stats

method = "UMAP"
dataset = "lish-moa"
clusters = pandas.read_csv(f"UMAP_analysis/f3_clustering/clusters/{method}_{dataset}.csv",index_col=0)
metadata = pandas.read_csv(f"data/intermediate/{dataset}/metadata.csv",index_col=0)
target = metadata.target

clusters = clusters[metadata.known==1]
target = target[metadata.known==1]

def cluster_dict(series):
    return {i:set(series[series == i].index) for i in series.unique()} 

def enrichment(set1, set2, n_total):
    return scipy.stats.hypergeom.pmf(len(set1&set2), n_total, len(set1), len(set2))

def cross_enrichment(dict1, dict2):
    out = {}
    for k1,v1 in dict1.items():
        out[k1]={}
        for k2,v2 in dict2.items():
            out[k1][k2] = enrichment(v1,v2,len(metadata.index))
    return pandas.DataFrame(out)

def bonferroni(enrichments):
    return enrichments * enrichments.size

out = {}
for key in clusters.keys():
    dict1 = cluster_dict(clusters[key])
    dict2 = cluster_dict(metadata.target)
    df = cross_enrichment(dict1, dict2)
    df = bonferroni(df)
    df.to_csv(f"UMAP_analysis/f3_clustering/enrichment/{dataset}/noise_{key}.csv")
    out[key] = (df<0.01).sum(axis=0)
