import cmapPy.pandasGEXpress.parse
import numpy as np
import pandas as pd

gct = cmapPy.pandasGEXpress.parse.parse("CCLE.gct", convert_neg_666=True)
features = gct.data_df


metadata = gct.row_metadata_df
metadata.reindex()
metadata.index = metadata.index.astype(float)
metadata.index = metadata.index.astype(int)



features.index = gct.row_metadata_df["compound_name"]
features -= 15.0
features = features.fillna(0.0)

names = metadata.compound_name.unique()


metadata.set_index("compound_name", drop = False, inplace=True)

tab = 0
out = []
for i in gct.col_metadata_df.cell_line_name.duplicated():
    
    if i:
        tab += 1
        out.append("_duplicate")
    else:
        tab = 0
        out.append("")
    assert tab <= 1
features.columns = gct.col_metadata_df.cell_line_name + np.array(out)


metadata#.to_csv("metadata.csv")
features#.to_csv("features.csv")
names# pd.Series(names).to_csv("names.csv", index=None)