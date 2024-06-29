# %%
import pandas as pd

names = ["gene", "value", "pred", "obs", "atpm"]
# %%
results_dir = "/burg/pmg/users/alb2281/get/results/gbm_deg_v1"

# %%
files = os.listdir(results_dir)

# %%
tumor_files = [item for item in files if "dea_Tumor" in item]
# %%

df_col = []
for file in files:
    df = pd.read_csv(f"{results_dir}/{file}", sep="\t", names=names)
    df_col.append(df)
    
# %%
fetal_celltype = "/burg/pmg/users/xf2217/get_revision/fetal_gene_expression_celltype.txt"


# %%
fetal_df = pd.read_csv(fetal_celltype, sep=",")
# %%
fetal_df
# %%
from caesar.io.zarr