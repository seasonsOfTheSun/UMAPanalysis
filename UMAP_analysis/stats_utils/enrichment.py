import scipy.stats
import pandas
import numpy


def cluster_dict(series):
    return {i:set(series[series == i].index) for i in series.unique()} 

def apply_to_cluster_dict(cluster_dict, f):
    out = {}
    for k in cluster_dict.keys():
        out[k] = f(cluster_dict[k])
    return out

def enrichment(set1, set2, n_total):
    return scipy.stats.hypergeom.pmf(len(set1&set2), n_total, len(set1), len(set2))


def pairwise_enrichment(clusters,target,n_total):
    dict1 = cluster_dict(clusters)
    dict2 = cluster_dict(target)
    dict2 = apply_to_cluster_dict(dict2, lambda x:x&set(target.index))
    N = len(set(target.index))
    out = {}
    for k1,v1 in dict1.items():
        out[k1]={}
        for k2,v2 in dict2.items():
            out[k1][k2] = enrichment(v1,v2,n_total)
    return pandas.DataFrame(out)


def pairwise_values(clusters,target,f):
    dict1 = cluster_dict(clusters)
    dict2 = cluster_dict(target)

    N = len(set(target.index))
    out = {}
    for k1,v1 in dict1.items():
        out[k1]={}
        for k2,v2 in dict2.items():
            out[k1][k2] = f(v1,v2)
    return pandas.DataFrame(out)


def safe_neg_log(df):
    min_ = df[df > 0].min().min()
    df = numpy.maximum(df, min_)
    return -numpy.log(df)


def pairwise_enrichment_adjusted(clusters,target, n_total):
    pvalues = pairwise_enrichment(clusters,target, n_total)
    neg_log_pvalues = safe_neg_log(pvalues)
    return neg_log_bonferroni(neg_log_pvalues)


def bonferroni(enrichments):
    return enrichments * enrichments.size

def neg_log_bonferroni(neg_log_enrichments):
    return neg_log_enrichments - numpy.log(neg_log_enrichments.size)

