# %%
import numpy as np 

# %%
data_dir = "/home/ubuntu/alb2281/get/output"
cols = ["gene", "value", "pred", "obs", "atpm"]

# %%
import os

sample_list = [item for item in os.listdir(data_dir) if item.startswith("gbm")]
# %%

import pandas as pd
oneshot = pd.read_csv("/home/ubuntu/alb2281/get/output/gbm_oneshot_gene_dea_Macrophages.htan_gbm.C3L-03405_CPT0224600013_snATAC_GBM_Tumor.512.csv", names=cols)
zeroshot = pd.read_csv("/home/ubuntu/alb2281/get/output/gbm_zeroshot_gene_dea_Macrophages.htan_gbm.C3L-03405_CPT0224600013_snATAC_GBM_Tumor.512.csv", names=cols)
# %%

import seaborn as sns
sns.scatterplot(data=zeroshot, x="obs", y="pred", s=1)
# %%
sns.scatterplot(data=oneshot, x="obs", y="pred", s=1)
# %%
sns.scatterplot(data=zeroshot, x="pred", y="obs", s=1)
# %%
sns.scatterplot(data=oneshot, x="pred", y="obs", s=1)
# %%
sns.scatterplot(data=zeroshot, x="pred", y="obs", s=1)
# %%

fetal_path = "/home/ubuntu/alb2281/get/output/fetal_gene_expression_celltype.txt"
fetal_df = pd.read_csv(fetal_path, sep=",")
# %%

from caesar.io.gencode import Gencode
# %%
gencode = Gencode()
# %%
gencode.gtf
# %%
# get dictionary mapping gene_id to gene_name using gene_id and gene_name columns
gene_id_to_name = gencode.gtf[["gene_id", "gene_name"]].drop_duplicates().set_index("gene_id")["gene_name"].to_dict()
# %%
fetal_df.rename(columns={"RowID": "gene_id"}, inplace=True)
fetal_df["gene_id"] = fetal_df["gene_id"].apply(lambda x: x.split(".")[0])
fetal_df["gene_name"] = fetal_df["gene_id"].apply(lambda x: gene_id_to_name[x] if x in gene_id_to_name else None)
# %%

fetal_df = fetal_df.dropna()

# %%
fetal_df.set_index("gene_name", inplace=True)

# %%
fetal_df.drop(columns=["gene_id"], inplace=True)
fetal_tpm = fetal_df.div(fetal_df.sum(axis=0), axis=1)
fetal_log_tpm = np.log10(1 + 1e6 * fetal_tpm)
# %%

gbm_samples = [item for item in os.listdir(data_dir) if item.startswith("gbm")]

# %%

# read all of the csvs into a dataframe
dfs = []
for sample in gbm_samples:
    path = os.path.join(data_dir, sample)
    df = pd.read_csv(path, names=cols)
    df["sample"] = sample
    dfs.append(df)
# %%
all_df = pd.concat(dfs)
# %%
all_df = all_df.set_index("gene")
# %%
all_df.rename(columns={"gene": "gene_name"}, inplace=True)
# %%
all_df.drop("value", inplace=True, axis=1)
# %%

# create dataframe where genes are rows and samples are columns and values are the observed values
all_df = all_df.pivot(columns="sample", values="obs")
# %%
all_df = all_df.dropna()
# %%
# rename index column
all_df.index.name = "gene_name"
# %%
# merge df on index
all_df = all_df.merge(fetal_log_tpm, left_index=True, right_index=True)
# %%

pairwise_corr = all_df.corr()
# %%
# get max value for each row from 54th column onwards and store the column name with the max 
# value in a new column
pairwise_corr["max_corr"] = pairwise_corr.iloc[:, 54:].idxmax(axis=1)


# %%
# store max value in a new column and exclude last column
pairwise_corr["max_corr_value"] = pairwise_corr.iloc[:, 54:-1].max(axis=1)
# %%

max_corr_df = pairwise_corr[["max_corr", "max_corr_value"]]
# %%
# only take first 54 rows
max_corr_df = max_corr_df.iloc[:54]

# %%

# set column to index
max_corr_df["sample"] = max_corr_df.index
max_corr_df["celltype"] = max_corr_df["sample"].apply(lambda x: x.split(".")[0].split("_")[-1])
# %%
max_corr_df["method"] = max_corr_df["sample"].apply(lambda x: x.split(".")[0].split("_")[1])
# %%

def get_case_id_from_sample(sample_id):
    parts = sample_id.split(".")
    celltype = parts[0].split("_")[-1]
    case_id = parts[2].split("_")[0]
    return f"{celltype}.{case_id}"

max_corr_df["case_id"] = max_corr_df["sample"].apply(get_case_id_from_sample)
# %%

max_corr_df = max_corr_df.set_index("case_id")

# %%
# keep columns with gbm in name
ground_truth = all_df.filter(regex="gbm_zeroshot")
ground_truth["Cerebrum-Astrocytes"] = all_df["Cerebrum-Astrocytes"]
ground_truth["Cerebrum-Microglia"] = all_df["Cerebrum-Microglia"]
ground_truth["Cerebrum-Oligodendrocytes"] = all_df["Cerebrum-Oligodendrocytes"]
 

# %%
merged_df = all_df.set_index("case_id")
# %%
stats_df = pd.merge(merged_df, max_corr_df, left_index=True, right_index=True)

# %%
import seaborn as sns
# %%
sns.scatterplot(data=stats_df, x="max_corr_value", y="exp_pearson", hue="method")
# %%
merged_df["case_id"] = merged_df.index
# %%
# melt dataframe merged_df
melted_df = pd.melt(merged_df, id_vars=["celltype", "patient_id", "case_id"], value_vars=["exp_pearson_zeroshot", "exp_pearson_oneshot"], var_name="method", value_name="exp_pearson")
# %%
melted_df.index = melted_df.case_id
# %%
max_corr_df_zeroshot = max_corr_df[max_corr_df["method"] == "zeroshot"]
max_corr_df_oneshot = max_corr_df[max_corr_df["method"] == "oneshot"]
# %%
melted_df_zeroshot = melted_df[melted_df["method"] == "exp_pearson_zeroshot"]
melted_df_oneshot = melted_df[melted_df["method"] == "exp_pearson_oneshot"]
# %%
merged_zeroshot = pd.merge(melted_df_zeroshot, max_corr_df_zeroshot, left_index=True, right_index=True)
# %%
merged_oneshot = pd.merge(melted_df_oneshot, max_corr_df_oneshot, left_index=True, right_index=True)
# %%
merged_zeroshot.drop(columns=["patient_id", "method_x", "sample", "celltype_y", "method_y"], inplace=True)
# %%
merged_zeroshot.rename(columns={"celltype_x": "celltype"}, inplace=True)
# %%
merged_oneshot.drop(columns=["patient_id", "method_x", "sample", "celltype_y", "method_y"], inplace=True)

# %%
merged_oneshot.rename(columns={"celltype_x": "celltype"}, inplace=True)
# %%

merged_zeroshot["method"] = "zeroshot"
merged_oneshot["method"] = "oneshot"
# %%
merged_zeroshot.rename(columns={"exp_pearson": "exp_pearson_zeroshot"}, inplace=True)
merged_oneshot.rename(columns={"exp_pearson": "exp_pearson_oneshot"}, inplace=True)
# %%

# %%
# concatenate columns
merged_final_df = merged_zeroshot.merge(merged_oneshot, on=["case_id", "celltype"], suffixes=("_zeroshot", "_oneshot"))
# %%

# %%
# drop teh column "case_id"
merged_zeroshot.drop(columns=["case_id"], inplace=True)
merged_oneshot.drop(columns=["case_id"], inplace=True)
# %%
merged_zeroshot = merged_zeroshot.reset_index()
merged_oneshot = merged_oneshot.reset_index()
# %%
final_merged = pd.merge(merged_zeroshot, merged_oneshot, on=["case_id"], suffixes=("_zeroshot", "_oneshot"))
# %%
final_merged = final_merged[["case_id", "celltype_zeroshot", "exp_pearson_zeroshot", "exp_pearson_oneshot", "max_corr_value_zeroshot", "max_corr_zeroshot"]]
# %%
final_merged = final_merged.rename(columns={"celltype_zeroshot": "celltype"})
final_merged = final_merged.rename(columns={"max_corr_value_zeroshot": "max_corr_value"})
final_merged = final_merged.rename(columns={"max_corr_zeroshot": "max_corr_celltype"})
# %%
final_merged = final_merged[final_merged["case_id"] != "Tumor.C3L-03405"]

# melt the df
final_merged = pd.melt(final_merged, id_vars=["celltype", "case_id", "max_corr_value", "max_corr_celltype"], value_vars=["exp_pearson_zeroshot", "exp_pearson_oneshot"], var_name="method", value_name="exp_pearson")
# %%
final_merged["method"] = final_merged["method"].apply(lambda x: x.split("_")[2])
# %%

# %%
# set axis from 0 to 1 for x and y

# %%

sns.scatterplot(data=final_merged, x="max_corr_value", y="exp_pearson", hue="method")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Max correlation with pretraining celltype")
plt.ylabel("Pearson correlation")
plt.title('GBM performance compared against pretraining set')
# Add a dotted line for y = x
x = np.linspace(0, 1, 100)
plt.plot(x, x, linestyle=':', color='black')
plt.show()
# %%

def preprocess_method_str(x):
    type = x.split("shot")[0]
    if type == "zero":
        return "Zero-shot"
    elif type == "one":
        return "One-shot (finetuned)"


# %%
final_merged["method"] = final_merged["method"].apply(preprocess_method_str)
# %%
sns.scatterplot(data=final_merged, x="max_corr_value", y="exp_pearson", hue="method")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Max correlation with pretraining celltype")
plt.ylabel("Pearson correlation")
plt.title('GBM performance compared against pretraining set')
# Add a dotted line for y = x
x = np.linspace(0, 1, 100)
plt.plot(x, x, linestyle=':', color='black')
plt.legend(title="")
plt.show()
# %%
sns.set_palette("Set2")
sns.scatterplot(data=final_merged, x="max_corr_value", y="exp_pearson", hue="celltype")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Max correlation with pretraining celltype")
plt.ylabel("Pearson correlation")
plt.title('GBM performance compared against pretraining set')
# Add a dotted line for y = x
x = np.linspace(0, 1, 100)
plt.plot(x, x, linestyle=':', color='black')
plt.legend(title="")
plt.show()
# %%
corr_by_type = final_merged[["case_id", "max_corr_value", "max_corr_celltype", "method"]]
corr_by_type = corr_by_type[corr_by_type["method"] == "Zero-shot"]
corr_by_type.to_csv("~/alb2281/get/results/corr_by_type.csv")
# %%

corr_by_type = corr_by_type.sort_values(by="max_corr_value", ascending=False)
corr_by_type.to_csv("~/alb2281/get/results/corr_by_type.csv")
# %%

corr_by_type["max_corr_value"] = corr_by_type["max_corr_value"].apply(lambda x: round(float(x), 3))
# %%
corr_by_type
# %%
corr_by_type.to_csv("~/alb2281/get/results/corr_by_type.csv")
# %%
all_df
# %%
# filter to first 54 columns
all_df = all_df.iloc[:, :54]
# %%
# rename column using lambda

def rename_column(x):
    parts = x.split(".")
    celltype = parts[0].split("_")[-1]
    case_id = parts[2].split("_")[0]
    return f"{celltype}.{case_id}"

all_df.columns = all_df.columns.map(rename_column)
# %%

# get list of most variable rows
most_variable = ground_truth.var(axis=1).sort_values(ascending=False).index[:100]
# %%
# filter to most variable rows
subset_df = ground_truth.loc[most_variable]


# %%
# rename columns 


def clean_col(x):
    if x.startswith("Cerebrum"):
        return x
    parts = x.split(".")
    celltype = parts[0].split("_")[-1]
    patient_id = parts[2].split("_")[0]
    return f"{celltype}.{patient_id}"

# rename the columns using lambda 
subset_df.columns = subset_df.columns.map(clean_col)

# %%
# clustermap of subset_df
import matplotlib.pyplot as plt 
# set figure size
plt.figure(figsize=(10, 10))
plt.title("Highly variable genes across all celltypes")
sns.clustermap(subset_df, cmap="viridis", figsize=(20, 20), method="ward")
plt.xlabel("Sample ID")
plt.ylabel("Gene name")
plt.show()

# %%
# filter to columns with Tumor in the name
tumor_df = all_df.filter(regex="Tumor")
# %%


# get list of most variable rows
most_variable = tumor_df.var(axis=1).sort_values(ascending=False).index[:100]
# %%
# filter to most variable rows
subset_tumor_df = tumor_df.loc[most_variable]

# clustermap of subset_df
# %%
sns.clustermap(subset_tumor_df, cmap="viridis")

# %%
print("test")
# %%



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_dir = "/home/ubuntu/alb2281/get/output"
# take everything after the second underscore
gbm_samples = [item.split("_", 2)[2] for item in os.listdir(data_dir) if item.startswith("gbm")]

# %%
def plot_sample(sample):
    parts = sample.split(".")
    celltype = parts[0].split("_")[-1]
    case_id = parts[2].split("_")[0]
    sample_id = f"{celltype}.{case_id}"
    zeroshot = pd.read_csv(f"{data_dir}/gbm_zeroshot_{sample}", names=cols)
    oneshot = pd.read_csv(f"{data_dir}/gbm_oneshot_{sample}", names=cols)
    # sns.scatterplot(data=oneshot, x="obs", y="pred", s=1)
    # show the two scatterplots side by side
    plt.figure()
    plt.title(f"{sample_id} zero-shot")
    sns.scatterplot(data=zeroshot, x="obs", y="pred", color="blue", s=1)
    plt.xlabel("Observed expression")
    plt.ylabel("Predicted expression")
    plt.savefig(f"/home/ubuntu/alb2281/plots/{sample_id}_zeroshot.png")

    plt.figure()
    plt.title(f"{sample_id} one-shot")
    sns.scatterplot(data=oneshot, x="obs", y="pred", color="blue", s=1)
    plt.xlabel("Observed expression")
    plt.ylabel("Predicted expression")
    plt.savefig(f"/home/ubuntu/alb2281/plots/{sample_id}_oneshot.png")

# %%

sns.set_palette("Set1")

# %%
for item in gbm_samples:
    plot_sample(item)

# %%
def process_sample_name(sample):
    parts = sample.split(".")
    method = parts[0].split("_")[1]
    celltype = parts[0].split("_")[-1]
    case_id = parts[2].split("_")[0]
    return f"{celltype}.{case_id}-{method}"



# %%
all_df["sample"] = all_df["sample"].apply(process_sample_name)
# %%

all_df["method"] = all_df["sample"].apply(lambda x: x.split("-")[1])

# %%
# melt the df by expanding method
melted_df = pd.melt(all_df, id_vars=["sample", "method"], value_vars=["obs", "pred"], var_name="value_type", value_name="value")