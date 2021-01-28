import pandas
import scipy.stats

method = "UMAP"
dataset = "cytodata"
clusters = pandas.read_csv(f"UMAP_analysis/f3_clustering/clusters/{method}_{dataset}.csv",index_col=0)
metadata = pandas.read_csv(f"munged_data/{dataset}/metadata.csv",index_col=0)
target = metadata.target

def cluster_dict(series):
    return {i:set(series[series == i].index) for i in series.unique()} 

def enrichment(set1, set2, n_total):
    return scipy.stats.hypergeom.pmf(len(set1&set2), n_total, len(set1), len(set2))


dict1 = cluster_dict(clusters['0.0'])   
dict2 = cluster_dict(metadata.target)
out = {}
for k1,v1 in dict1.items():
    for k2,v2 in dict2.items():
        out[(k1, k2)] = enrichment(v1,v2)

enrichment(cluster_dicts(clusters['0.5'])[17]
