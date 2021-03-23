import pandas as pd
import sys
dataset = sys.argv[1]
df = pd.read_csv(f"data/intermediate/{dataset}/labels.csv", index_col = 0)

for i,moa in enumerate(df.columns):
    fp = open(f"data/intermediate/{dataset}/classes/class_{i}", 'w')
    fp.write(moa)
    fp.close()
