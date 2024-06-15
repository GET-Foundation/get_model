# %%
debug = "/pmglocal/alb2281/get/output/watac-zeroshot-astrocyte/gbm_zeroshot_gene_dea_Tumor.htan_gbm.C3L-03405_CPT0224600013_snATAC_GBM_Tumor.4096_watac.csv"
import pandas as pd
# %%
names = ["gene", "val", "pred", "obs", "atpm"]
df = pd.read_csv(debug, names=names)
# %%
import seaborn as sns
sns.scatterplot(data=df, x="obs", y="pred", s=1)
# %%
