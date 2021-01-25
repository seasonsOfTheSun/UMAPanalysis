
import matplotlib.pyplot as plt
import pandas as pd
import os
import re

def make_roc_fig(moa):
    df = pd.read_csv("performance_metrics/moa_csvs/" + moa + "_roc_curves.csv", index_col=0)
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    ax.plot(df.propagation_FPR, df.propagation_TPR, c = 'k')
    ax.plot(df.random_forest_FPR, df.random_forest_TPR, c = '#00c2d7')
    ax.plot(df.nearest_neighbor_FPR, df.nearest_neighbor_TPR, c =  '#ae342b')
    ax.set_title(" ".join([x.title() for x in moa.split("_")]))
    ax.set_xlabel("False Positive Rate", fontsize = 10)
    ax.set_ylabel("True Positive Rate",  fontsize = 10)

    return fig

files = os.listdir("performance_metrics/moa_csvs/")

for file in files:
    m = re.match("(.*?)_roc_curves.csv", file)
    if m is not None:
        moa = m.groups()[0]
        fig = make_roc_fig(moa)
        fig.savefig("figures/roc_curves/"+moa+".svg")
        plt.close(fig)