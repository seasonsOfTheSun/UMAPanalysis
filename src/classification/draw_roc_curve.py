import re
import sys
from UMAP_analysis.stats_utils.receiver_operating_characteristic import *

m = re.match("predictions/(?P<dataset>.*?)/(?P<prediction_method>.*?).csv", sys.argv[1])
dataset = m.groupdict()['dataset']
prediction_method = m.groupdict()['prediction_method']
moa = sys.argv[2]


truth = pd.read_csv(f"data/intermediate/{dataset}/labels.csv", index_col=0)
prediction = pd.read_csv(f"predictions/{dataset}/{prediction_method}.csv", index_col=0)


fig = make_roc_fig(moa)
fig.savefig("figures/roc_curves/{dataset}/{prediction_method}_"+moa+".svg")
plt.close(fig)
