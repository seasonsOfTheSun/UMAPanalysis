


# Download morphological data http://gigadb.org/dataset/100351

import os
import pandas as pd

drug_df = pd.read_csv("chemical_annotations.csv").set_index("BROAD_ID")
to_type = (drug_df.CPD_NAME_TYPE).to_dict().get

plates = os.listdir("profiles.dir/")
out = []
for plate in plates:
    df = pd.read_csv(f"profiles.dir/{plate}/profiles/mean_well_profiles.csv")
    out.append(df[~(df.Metadata_broad_sample.map(to_type) == 'BROAD_CPD_ID')])
morpho_df = pd.concat(out)
#morpho_df.to_csv("known_drug_morphology.csv")

#morpho_df = pd.read_csv("known_drug_morphology.csv", index_col = 0)


morpho_df["Metadata_pert_mfc_id"] = morpho_df["Metadata_pert_mfc_id"].fillna("DMSO")
morpho_df = morpho_df.set_index("Metadata_pert_mfc_id", drop=False)
morpho_df.index = [drug if drug != 'DMSO' else drug + "_" + str(n)  for n,drug in enumerate(morpho_df.index)]


drug_df_select = drug_df[~(drug_df.CPD_NAME_TYPE == 'BROAD_CPD_ID')]
names = drug_df_select[["CPD_NAME", "CPD_SMILES"]]


import re
import numpy as np


metadata = [i for i in morpho_df.columns if re.match("Metadata", i)]
numeric = [i for i in morpho_df.columns if not re.match("Metadata", i)]

for constant_feature in np.array(numeric)[morpho_df[numeric].std() == 0]:
    numeric.remove(constant_feature)

features = morpho_df[numeric]/morpho_df[numeric].std()

metadata = morpho_df[metadata].join(drug_df_select)


features.to_csv("data/intermediate/morphological/features.csv")
metadata.to_csv("data/intermediate/morphological/metadata.csv")
pd.Series(metadata.CPD_NAME.dropna().unique()).to_csv("data/intermediate/morphological/drug_names.csv", index = None)

### features_unscaled.to_csv("data/intermediate/morphological/features_unscaled.csv")
### y.to_csv("data/intermediate/morphological/labels.csv")
### names.to_csv("morpho_drug_name.csv")
