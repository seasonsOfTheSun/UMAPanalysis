import cmapPy.pandasGEXpress.parse

data = cmapPy.pandasGEXpress.parse.parse_gctx.parse("LDS-1194/Data/zspc_n70323x22268.gctx")

features = data.data_df.transpose()



landmark_features = data.row_metadata_df.query("pr_is_lmark == 'Y'").index

assert not data.row_metadata_df.loc[landmark_features].pr_gene_symbol.duplicated().any()

features = features[landmark_features]

features.columns = data.row_metadata_df.loc[landmark_features].pr_gene_symbol.values

features = features/features.std()

assert not features.isna().any().any()



data.col_metadata_df.columns


names = data.col_metadata_df.pert_desc

metadata = data.col_metadata_df

features.to_csv("data/intermediate/morphological/features.csv")
metadata.to_csv("data/intermediate/morphological/metadata.csv")
pd.Series(names).to_csv("data/intermediate/morphological/drug_names.csv", index = None)
