import collections
import sklearn.metrics

import os
os.chdir('/Users/jhancock/UMAP_analysis/')

def shannon_entropy(labels):
    n = len(labels)
    c = collections.Counter(labels.values) # get counts for each label.
    out = 0
    for count in c.values(): # iterate across labels.
        p = count/n # probability of having specific label.
        out += -p*np.log(p) # get summands in entropy formula.
    return out

def variation_of_information(labels_true, labels_pred):
    entropy_true = shannon_entropy(labels_true)
    entropy_pred = shannon_entropy(labels_pred)
    mutual_information = sklearn.metrics.mutual_info_score(labels_true, labels_pred)
    return entropy_true + entropy_pred - 2*mutual_information