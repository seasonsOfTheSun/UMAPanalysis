import sys
import pandas as pd
import numpy as np

filename = sys.argv[1]
evector = pd.read_csv(filename, index_col = 0)
dataset = filename.split("/")[2]

network_filename = sys.argv[1]
m = re.match("networks/(?P<dataset>.*?)/eigenvectors/(?P<name>.*?).csv", network_filename)
dataset = m.groupdict()['dataset']
name = m.groupdict()['name']

evec_keys = sorted(evector.columns, key=lambda i:evector.loc["lambda",i])
n = evec_keys[-2]
fiedler = evector[n]
del fiedler['lambda']


filename_out = "fiedler_"+name+".csv"

def closest_integer(x):
    if x >= 0.5:
       return 1
    else:
       return 0
median = fiedler.median()
df = (fiedler > median).astype(int)
df.to_csv(f"data/processed/clusters/{dataset}/median_" + filename_out, header = False)

df = (fiedler > 0).astype(int)
df.to_csv(f"data/processed/clusters/{dataset}/sign_" + filename_out, header = False) 
