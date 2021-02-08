import sys
import pandas as pd
import numpy as np

filename = sys.argv[1]
evector = pd.read_csv(filename, index_col = 0)
dataset = filename.split("/")[-3]

n = np.argmax(evector.loc["lambda", evector.loc["lambda"] < 1])
fiedler = evector[n]
del fiedler['lambda']


filename_out = "fiedler_"+filename.split("/")[-1].split(".")[0]+".csv"

def closest_integer(x):
    if x >= 0.5:
       return 1
    else:
       return 0
median = fiedler.median()
df = (fiedler > median).astype(int)
df.to_csv(f"data/processed/clusters/{dataset}/"+filename_out, header = False)
