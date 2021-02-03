import pandas as pd
import numpy as np
import os
os.chdir('//')

dataset = "data/raw/lish-moa"

for dataset in ['lish-moa', 'cytodata', 'GDSC']:
    metadata = pd.read_csv(f"data/intermediate/{dataset}/metadata.csv",  index_col = 0)

    to_choose_from = metadata.index[metadata.known == 1]
    test = np.random.choice(to_choose_from, int(len(to_choose_from)/3), replace=False)
    train = [i for i in to_choose_from if i not in test]

    pd.Series(test).to_csv(f"4.classification/train_and_test/{dataset}/test.csv", index=None, header = False)
    pd.Series(train).to_csv(f"4.classification/train_and_test/{dataset}/train.csv", index=None, header = False)
