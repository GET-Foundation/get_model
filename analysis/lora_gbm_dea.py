# %%
import pandas as pd 

# %%
names = ["gene", "val", "pred", "obs", "atpm"]
zeroshot_output = "/pmglocal/alb2281/get/output/gbm_zeroshot_gene_dea_Tumor.htan_gbm.C3L-03405_CPT0224600013_snATAC_GBM_Tumor.4096_natac.csv"
zeroshot_output = pd.read_csv(zeroshot_output, names=names)

# %%
import seaborn as sns
# %%
sns.scatterplot(data=zeroshot_output, x="obs", y="pred", s=1)
# %%
