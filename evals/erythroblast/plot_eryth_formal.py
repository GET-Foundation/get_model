
# %%
import hydra
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from sklearn.metrics import average_precision_score, precision_recall_curve
import torch
import zarr
import numpy as np
from pyranges import PyRanges as prs , read_bed
from get_model.utils import rename_lit_state_dict
#%%
# DistanceMap/kp6cfqkc/checkpoints/best.ckpt
config = OmegaConf.load(
    '/home/xf2217/Projects/get_model/get_model/config/model/DistanceContactMap.yaml')
model = hydra.utils.instantiate(config)['model']
checkpoint = torch.load(
    '/home/xf2217/Projects/get_model/DistanceMap/kp6cfqkc/checkpoints/best.ckpt')
model.load_state_dict(rename_lit_state_dict(checkpoint['state_dict']))
#%%
z = zarr.open('/home/xf2217/output/GETRegionFinetuneV1_Erythroblast/eryth.zarr', 'r')
print(z.tree())
#%%
gene_names = z.gene_name[:]
strands = z.strand[:].astype(int)
peak_coord = z.peak_coord[:].astype(int)
chromosome = z.chromosome[:]
obs = z.obs.exp[:]
preds = z.pred.exp[:]
input = z.input[:]
#%%
peak_coord = torch.Tensor(peak_coord)
peak_length = peak_coord[:, :, 1] - peak_coord[:, :, 0]
peak_coord_mean = peak_coord[:, :, 0]
# pair-wise distance using torch
# Add new dimensions to create column and row vectors
peak_coord_mean_col = peak_coord_mean.unsqueeze(2)  # Adds a new axis (column vector)
peak_coord_mean_row = peak_coord_mean.unsqueeze(1)  # Adds a new axis (row vector)

# Compute the pairwise difference
distance = torch.log10(
    (peak_coord_mean_col - peak_coord_mean_row).abs() + 1).unsqueeze(1)
# forward through the model
predicted_hic = model(distance).squeeze(1).detach().cpu().numpy()[:, 450, :]
#%%
def extract_expression_jacobian(z, layer='input'):
    strands = z.strand[:]
    positive_strand_jacobian = z.jacobians.exp['0'][layer][:]
    negative_strand_jacobian = z.jacobians.exp['1'][layer][:]
# get corresponding strands
    jacobians = []
    for i, strand in enumerate(strands):
        if strand == 0:
            jacobians.append(positive_strand_jacobian[i])
        else:
            jacobians.append(negative_strand_jacobian[i])

    jacobians = np.stack(jacobians)
    return jacobians

jacobians_input = extract_expression_jacobian(z, 'input')
jacobians_region_embed = extract_expression_jacobian(z, 'region_embed')
#%%
#%%
from pyliftover import LiftOver
lo = LiftOver('hg19', 'hg38')
lor = LiftOver('hg38', 'hg19')
#%%
import pandas as pd
annot = []
gene_to_tss = {}
gene_to_tss_hg38 = {}
for i, peak_coord_i in enumerate(peak_coord):
    annot_i = pd.DataFrame(peak_coord_i, columns=['Start', 'End']).astype(int)
    annot_i['Chromosome'] = chromosome[i].strip(' ')
    annot_i['Observation'] = obs[i][:, strands[i]]
    annot_i['Prediction'] = preds[i][:, strands[i]]
    annot_i['Gene'] = gene_names[i].strip(' ')
    annot_i['Strand'] = strands[i]
    annot_i['ATAC'] = input[i][:, 282]
    annot_i['Distance'] = np.abs(annot_i['Start'] - annot_i.iloc[450]['Start'])
    annot_i['predicted_hic'] = predicted_hic[i]
    annot_i['jacobian_norm'] = np.linalg.norm(jacobians_region_embed[i], axis=1)
    annot.append(annot_i)
    gene_to_tss[gene_names[i].strip(' ')] = annot_i.iloc[450]['Start']
    gene_to_tss_hg38[gene_names[i].strip(' ')] = lo.convert_coordinate(chromosome[i].strip(' '), annot_i.iloc[450]['Start'])[0][1]
annot = pd.concat(annot)
annot['ID'] = annot.Chromosome + ':' + annot.Start.astype(str) + '-' + annot.End.astype(str) + ':' + annot.Gene
#%%
annot.plot(kind='scatter', x='Observation', y='ATAC')

#%%
from pyranges import read_bigwig, read_bed
atac_bw = prs(read_bigwig("atac.hg38.bw").as_df().rename({'Value':'HUDEP ATAC'}, axis=1), int64=True)
#%%
annot_hg38 = annot.reset_index()[['Chromosome', 'Start', 'End', 'ATAC']].copy()
annot_hg38['Chromosome'] = annot_hg38['Chromosome'].str.strip(' ')
#%%
for i, row in annot_hg38.iterrows():
    annot_hg38.loc[i, 'Start'] = lo.convert_coordinate(row['Chromosome'], row['Start'])[0][1]
    annot_hg38.loc[i, 'End'] = lo.convert_coordinate(row['Chromosome'], row['End'])[0][1]


# strip chromosome
#%%
ABE = pd.read_csv("Editable_A_scores.combined.scores.csv")

# get Chromosome, Start, End, Ref, Alt from coord: chr11:4167374-4167375+, if coord ends with +, then Ref is A and Alt is G, if coord ends with -, then Ref is T and Alt is C
ABE['Chromosome'] = ABE['coord'].apply(lambda x: x.split(':')[0])
ABE['Start'] = ABE['coord'].apply(lambda x: int(x.split(':')[1].split('-')[0]))
ABE['End'] = ABE['coord'].apply(lambda x: int(x.split(':')[1].split('-')[1].split('+')[0]))
ABE['Ref'] = ABE['coord'].apply(lambda x: 'A' if x.endswith('+') else 'T')
ABE['Alt'] = ABE['coord'].apply(lambda x: 'G' if x.endswith('+') else 'C')
ABE 
#%%
ABE_hg38 = ABE.copy()
for i, row in ABE.iterrows():
    ABE_hg38.loc[i, 'Start'] = lo.convert_coordinate(row['Chromosome'], row['Start'])[0][1]
    ABE_hg38.loc[i, 'End'] = lo.convert_coordinate(row['Chromosome'], row['End'])[0][1]
#%%
# join ABE and annot
from pyranges import PyRanges as prs
ABE_annot = prs(annot, int64=True).join(prs(ABE, int64=True)).df
ABE_jacob_hg19 = ABE_annot[['ID', 'Distance', 'predicted_hic', 'Observation', 'Prediction', 'jacobian_norm', 'ATAC', 'DeepSEA', 'GERP', 'HbFBase', 'CADD']].groupby('ID').max()
#%%
ABE_jacob_hg19.plot(kind='scatter', x='HbFBase', y='jacobian_norm')
#%%
ABE_jacob_hg19['Chromosome'] = ABE_jacob_hg19.index.str.split(':').str[0]
ABE_jacob_hg19['Start'] = ABE_jacob_hg19.index.str.split(':').str[1].str.split('-').str[0].astype(int)
ABE_jacob_hg19['End'] = ABE_jacob_hg19.index.str.split(':').str[1].str.split('-').str[1].astype(int)
ABE_jacob_hg19['Gene'] = ABE_jacob_hg19.index.str.split(':').str[2]
ABE_jacob_hg38 = ABE_jacob_hg19.copy()
#%%
for i, row in ABE_jacob_hg38.iterrows():
    ABE_jacob_hg38.loc[i, 'Start'] = lo.convert_coordinate(row['Chromosome'], row['Start'])[0][1]
    ABE_jacob_hg38.loc[i, 'End'] = lo.convert_coordinate(row['Chromosome'], row['End'])[0][1]
#%%
def get_powerlaw_at_distance(distances, min_distance=5000):
    gamma = 1.024238616787792
    scale = 5.9594510043736655

    # The powerlaw is computed for distances > 5kb. We don't know what the contact freq looks like at < 5kb.
    # So just assume that everything at < 5kb is equal to 5kb.
    # TO DO: get more accurate powerlaw at < 5kb
    distances = np.clip(distances, min_distance, np.Inf)
    log_dists = np.log(distances + 1)

    powerlaw_contact = np.exp(scale + -1 * gamma * log_dists)
    return powerlaw_contact

ABE_jacob_hg38['Powerlaw'] = get_powerlaw_at_distance(ABE_jacob_hg38['Distance'])

# %%
ABE_jacob_hg38
# %%
ABE_jacob_hg38_enformer=[]
for g in ['MYB', 'BCL11A', 'NFIX', 'HBG2']:
    ABE_jacob_hg38_g = ABE_jacob_hg38.query(f'Gene=="{g}"')
    enformer_score = np.load("cage_cd34_diff_rbc." + g.lower() + ".contribution_scores.npy")
    center_len = 196608
    enformer_score = enformer_score[len(enformer_score)//2-center_len//2:len(enformer_score)//2+center_len//2]
    chrom = ABE_jacob_hg38_g['Chromosome'].values[0]
    start = gene_to_tss_hg38[g]
    track_start = start - 196608//2
    track_end = start + 196608//2
    track_abs_mean = np.abs(enformer_score).mean()
    df_subset = ABE_jacob_hg38_g
    print(df_subset.shape)
    enformer_score_result = []
    for i, row in df_subset.iterrows():
        peak_mid_point_in_track = (row['Start']+row['End'])//2-track_start
        peak_start_in_track = peak_mid_point_in_track-1000
        peak_end_in_track = peak_mid_point_in_track+1000

        if peak_start_in_track < 0 or peak_end_in_track > 196608:
            enformer_score_result.append(0)
        else:
            peak_score = np.abs(enformer_score[peak_start_in_track:peak_end_in_track]).mean()/track_abs_mean
            enformer_score_result.append(peak_score)
    df_subset['Enformer'] = enformer_score_result
    ABE_jacob_hg38_enformer.append(df_subset)

ABE_jacob_hg38_enformer = pd.concat(ABE_jacob_hg38_enformer)
ABE_jacob_hg38_enformer
# %%
hyena = pd.read_csv("./hyena_aggregate_erythro_900region.tsv", sep='\t')
hyena.rename(columns={'chrom':'Chromosome', 'chromStart':'Start',
                      'chromEnd':'End', 'mean_score_elementwise_abs':'hyena'}, inplace=True)
hyena  = hyena[['Chromosome', 'Start', 'End', 'hyena']]

# %%
np.bool = np.bool_
ABE_jacob_hg38_enformer_hyena = prs(ABE_jacob_hg38_enformer.reset_index(), int64=True).join(prs(hyena, int64=True), how='left').df.drop(columns=['Start_b', 'End_b'])
ABE_jacob_hg38_enformer_hyena['hyena'][ABE_jacob_hg38_enformer_hyena['hyena']==-1] = 0
ABE_jacob_hg38_enformer_hyena = ABE_jacob_hg38_enformer_hyena.groupby('ID').max()
# %%
ABE_jacob_hg38_enformer_hyena['ABC Powerlaw'] = ABE_jacob_hg38_enformer_hyena['ATAC'] * ABE_jacob_hg38_enformer_hyena['Powerlaw']
#%%
def normalize_arr(arr):
    return arr/arr.sum()

final_result = []
# normalize and compute GET
for g in ['MYB', 'BCL11A', 'NFIX', 'HBG2']:
    A_i = ABE_jacob_hg38_enformer_hyena.query(f'index.str.contains("{g}")')
    A_i['GET (Jacobian, ATAC, Powerlaw)'] = normalize_arr(A_i['jacobian_norm'] * A_i['ATAC'] + A_i['ATAC'] * A_i['predicted_hic'])
    A_i['GET (Jacobian, Powerlaw)'] = normalize_arr(A_i['jacobian_norm'] * A_i['predicted_hic'])
    A_i['GET (Jacobian)'] = normalize_arr(A_i['jacobian_norm'] * A_i['ATAC'])
    A_i['ABC Powerlaw'] = normalize_arr(A_i['ABC Powerlaw'])
    A_i['Enformer (Input x Attention)'] = normalize_arr(A_i['Enformer'])
    A_i['ATAC'] = normalize_arr(A_i['ATAC'])
    A_i['HyenaDNA (ISM)'] = normalize_arr(A_i['hyena'])
    A_i['DeepSEA (ISM)'] = normalize_arr(A_i['DeepSEA'])
    final_result.append(A_i)

final_result = pd.concat(final_result)

#%%
final_result['Regulated'] = final_result['HbFBase'] > final_result['HbFBase'].quantile(0.75)

# %%
import numpy as np
import seaborn as sns

def plot_aupr_bar(final_result, lower=None, upper=None, ax=None, n_resamples=1000, resample_size=0.8):
    methods = ['GET (Jacobian, ATAC, Powerlaw)', 'GET (Jacobian)', 'GET (Jacobian, Powerlaw)',  'Enformer (Input x Attention)', 'HyenaDNA (ISM)',  'DeepSEA (ISM)', 'ABC Powerlaw',  'ATAC', 'Distance']
    average_precisions = {}
    std_devs = {}
    ci95 = {}

    for method in methods:
        if method == 'Distance':
            scores = 1 / (final_result['Distance']+10)
        else:
            scores = final_result[method]


        if n_resamples>1:
            average_precisions[method] = []
            
            for _ in range(n_resamples):
                resampled_indices = np.random.choice(len(final_result), size=int(len(final_result) * resample_size), replace=False)
                resampled_overlap = final_result.iloc[resampled_indices]
                resampled_scores = scores.iloc[resampled_indices]

                average_precision = average_precision_score(resampled_overlap['Regulated'], resampled_scores)
                average_precisions[method].append(average_precision)

        else:
            
            average_precision = average_precision_score(final_result['Regulated'], scores)
            average_precisions[method]=average_precision
    if n_resamples>1:
        for method in methods:
            std_devs[method] = np.std(average_precisions[method])
            # quantile 95% confidence interval
            ci95[method] = np.quantile(average_precisions[method], [0.025, 0.975])
            average_precisions[method] = np.mean(average_precisions[method])
            
    if ax is None:
        ax = plt.gca()

    labels = methods#['GET (Jacobian x DNase, Powerlaw)', 'GET (Jacobian x ATAC)', 'GET (Jacobian, Powerlaw)', 'Enformer (Input x Attention)', 'HyenaDNA (ISM)',  'ABC Powerlaw', 'DNase', 'Distance']
    values = [average_precisions[method] for method in methods]
    
    if n_resamples>1:
        errors = [ci95[method][1] - average_precisions[method] for method in methods]
    else:
        errors = [0 for method in methods]
    # add random
    labels.append('Random')
    values.append(final_result['Regulated'].mean())
    errors.append(0)


    sns.barplot(x=labels, y=values, ax=ax, hue=labels, dodge=False, hue_order=labels)
    if n_resamples>1:
        ax.errorbar(x=range(len(labels)), y=values, yerr=errors, fmt='none', capsize=5, ecolor='black')
    n_positive = final_result['Regulated'].sum()
    n_negative = len(final_result) - n_positive
    if lower is not None and upper is not None:
        ax.set_xlabel('[{},{})bp\nPositive:{}\nNegative:{}'.format(int(lower), int(upper), int(n_positive), int(n_negative)))
    # rotate x labels
    ax.set_xticklabels("")
    
    # use legend instead of ticks
    ax.legend()
    # ax.set_ylim(0, 1)
#%%
final_result['Regulated'] = final_result['HbFBase'] > final_result['HbFBase'].quantile(0.75)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes = axes.flatten()
distances = [0, 100000]

# plot the total data in ax[0]
# plot_aupr_bar(overlap.dropna(), lower=0, upper=2.5e6, ax=axes[0])

for i, distance in enumerate(distances):
    o = final_result.dropna().query('Distance>' + str(distance) + ' & Distance<=' + str(distances[i+1] if i+1<len(distances) else 1e6))
    plot_aupr_bar(o, lower=distance, upper=distances[i+1] if i+1<len(distances) else 5e6, ax=axes[i])

# remove ax[0,1] legends
axes[0].legend().remove()
axes[1].legend().remove()
# put legend below figure in the middle with out affecting the panels
axes[0].legend(loc='upper center', bbox_to_anchor=(1, -0.3), ncol=4)

axes[0].set_title('AUPRC vs distance to TSS')
# save to pdf
# plt.tight_layout()
plt.savefig('auprc_vs_distance_0.75.pdf')
# %%
# plot 0.9 quantile
final_result['Regulated'] = final_result['HbFBase'] > final_result['HbFBase'].quantile(0.9)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes = axes.flatten()
distances = [0, 100000]

# plot the total data in ax[0]
# plot_aupr_bar(overlap.dropna(), lower=0, upper=2.5e6, ax=axes[0])

for i, distance in enumerate(distances):
    o = final_result.dropna().query('Distance>' + str(distance) + ' & Distance<=' + str(distances[i+1] if i+1<len(distances) else 1e6))
    plot_aupr_bar(o, lower=distance, upper=distances[i+1] if i+1<len(distances) else 5e6, ax=axes[i])

# remove ax[0,1] legends
axes[0].legend().remove()
axes[1].legend().remove()
# put legend below figure in the middle with out affecting the panels
axes[0].legend(loc='upper center', bbox_to_anchor=(1, -0.3), ncol=4)

axes[0].set_title('AUPRC vs distance to TSS')
# save to pdf
# plt.tight_layout()
plt.savefig('auprc_vs_distance_0.9.pdf')
# %%
# plot gene
final_result['Regulated'] = final_result['HbFBase'] > 30# final_result['HbFBase'].quantile(0.9)

fig, axes = plt.subplots(2, 2, figsize=(15, 8))
axes = axes.flatten()

for i, g in enumerate(['MYB', 'BCL11A', 'NFIX', 'HBG2']):
    o = final_result.dropna().query('index.str.contains(@g)')
    print(g)
    plot_aupr_bar(o, ax=axes[i], n_resamples=1, resample_size=1)
    axes[i].set_title(g)

# remove ax[0,1] legends
axes[0].legend().remove()
axes[1].legend().remove()
axes[2].legend().remove()
axes[3].legend().remove()
# put legend below figure in the middle with out affecting the panels
axes[2].legend(loc='upper center', bbox_to_anchor=(1, -0.3), ncol=4)

# save to pdf
# plt.tight_layout()
plt.savefig('auprc_vs_gene_0.9.pdf')
# %%
# plot gene
final_result['Regulated'] = final_result['HbFBase'] > 30 #final_result['HbFBase'].quantile(0.90)

fig, axes = plt.subplots(2, 2, figsize=(15, 9))
axes = axes.flatten()

for i, g in enumerate(['MYB', 'BCL11A', 'NFIX', 'HBG2']):
    o = final_result.dropna().query('index.str.contains(@g) & Distance>100000')
    print(g)
    plot_aupr_bar(o, ax=axes[i], n_resamples=1, resample_size=1)
    axes[i].set_title("Distance > 100kb\n" + g + ", Positive: " + str(o['Regulated'].sum()) + ", Negative: " + str(len(o)-o['Regulated'].sum()))

# remove ax[0,1] legends
axes[0].legend().remove()
axes[1].legend().remove()
axes[2].legend().remove()
axes[3].legend().remove()
# put legend below figure in the middle with out affecting the panels
axes[2].legend(loc='upper center', bbox_to_anchor=(1, -0.3), ncol=4)

# save to pdf
# plt.tight_layout()
plt.savefig('auprc_vs_gene_0.9_distal.pdf')
# %%
import matplotlib as mpl
def plot_bedgraph(df, ax, color, label, score = 'Score', absolute=False, dot=True, log=False):
    df = df.copy().sort_values('Start')
    df['mid'] = (df['Start']+df['End'])/2
    df['Start'] = df['mid']
    df['End'] = df['mid']+1
    df = df[['Chromosome', 'Start', 'End', score]]
    if absolute:
        df[score] = np.abs(df[score])
    if log:
        df[score] = np.log(df[score]+1)
    df[score] = (df[score])/(df[score].max())
    if dot:
        g = sns.scatterplot(data=df.query(f'`{score}`>0.02'), x='Start', y=score, ax=ax, color=color, label=label)
    else:
        g = sns.scatterplot(data=df.query(f'`{score}`>0.02'), x='Start', y=score, ax=ax, color=color, label=label, alpha=0)
    # rotate 90 degree, align to the right, put in the middle
    
    # add vertical lines for each scatter, height is the score
    for i, row in df.iterrows():
        g.vlines(row['Start'], 0, row[score], color=color, alpha=0.5)
    g.set_ylabel(label, rotation=0, ha='right', va='center')
    return g

def filter_hichip_by_tss(chrom, tss_start, hichip):
    hichip_filtered = hichip.query(f'Chromosome1=="{chrom}" & {tss_start}-Start1<4000000 & End2-{tss_start}<4000000')
    return hichip_filtered.query('Pval<0.5')

# def plot_gene_tss(cell, gene, ax, color, label, score = 'exp', bar_length = 10000):
#     tss_list = cell.get_gene_tss(gene)
#     score_list = []
#     if score == 'exp':
#         score_list = cell.get_gene_pred(gene)
#     elif score == 'atac':
#         score_list = cell.get_gene_accessibility(gene).toarray().flatten()
#     start_list = [tss.start for tss in tss_list]
#     print(start_list)
#     # add a bar with width 200 and height score around each start
#     for start, score in zip(start_list, score_list):
#         print(start, score)
#         ax.bar(start, score, width=bar_length, color=color, label=label)
#     return ax

hichip = pd.read_csv('hichip.bedpe', sep='\t', header=None, names = ['Chromosome1', 'Start1', 'End1', 'Chromosome2', 'Start2', 'End2', 'Count', 'Pval'])

from matplotlib.patches import Arc
def plot_bedpe(df, start, end,  ax, color, labels):
    for index, row in df.iterrows():
        start1 = row['Start1']
        end1 = row['End1']
        start2 = row['Start2']
        end2 = row['End2']
        c = row['Count']
        
        x1 = start1 + (end1 - start1) / 2  # X coordinate for the middle of region 1
        x2 = start2 + (end2 - start2) / 2  # X coordinate for the middle of region 2

        xy = (x1 + x2) / 2  # X coordinate for the middle between the two regions
        width = abs(x2 - x1)  # Distance between the two regions

        # use arc
        arc = Arc(xy=(xy, 0), width=width, height=2, theta1=0, theta2=360, edgecolor=color, lw=1, linestyle='-', fill=False, alpha=min(1, c/5))
        ax.add_patch(arc)


    ax.set_ylim(-1.2,0)
    # hide y ticks
    ax.set_yticks([])
    ax.set_xlim(start, end)
#%%
final_result_hg38_coord = final_result.copy()
final_result_hg38_coord['Chromosome'] = final_result_hg38_coord.index.str.split(':').str[0]
final_result_hg38_coord['Start'] = final_result_hg38_coord.index.str.split(':').str[1].str.split('-').str[0].astype(int)
final_result_hg38_coord['End'] = final_result_hg38_coord.index.str.split(':').str[1].str.split('-').str[1].astype(int)
final_result_hg38_coord['Gene'] = final_result_hg38_coord.index.str.split(':').str[2]
for i, row in final_result_hg38_coord.iterrows():
    final_result_hg38_coord.loc[i, 'Start'] = lo.convert_coordinate(row['Chromosome'], row['Start'])[0][1]
    final_result_hg38_coord.loc[i, 'End'] = lo.convert_coordinate(row['Chromosome'], row['End'])[0][1]
#%%

def plot_gene_crispr(gene, final_result, coord=(100000, 100000), score='GET (Jacobian)'):
    df = final_result.query(f'index.str.contains("{gene}")').copy().fillna(0)
    chrom = df['Chromosome'].values[0]
    tss_start = gene_to_tss_hg38[gene]
    fig, ax = plt.subplots(nrows=8, ncols=1, figsize=(10, 5), sharex=True, gridspec_kw={'height_ratios': [2, 2,2,2, 2,2,2,2], 'hspace': 0.3})
    plot_bedgraph(ABE_hg38.query(f'Chromosome=="{chrom}" & Start>{str(coord[0])} & End<{str(coord[1])}'), ax[0], '#b3b3b3', 'HbFBase', score='HbFBase')
    plot_bedgraph(df.query(f'Chromosome=="{chrom}" & Start>{str(coord[0])} & End<{str(coord[1])}'), ax[1], '#E2812D', 'GET (Jacobian)', score=score, absolute=False)
    plot_bedgraph(df.query(f'Chromosome=="{chrom}" & Start>{str(coord[0])} & End<{str(coord[1])}'), ax[2], '#C33D3F', 'Enformer', score='Enformer', absolute=False)
    plot_bedgraph(df.query(f'Chromosome=="{chrom}" & Start>{str(coord[0])} & End<{str(coord[1])}'), ax[3], '#9273B3', 'HyenaDNA', score='HyenaDNA' )
    plot_bedgraph(df.query(f'Chromosome=="{chrom}" & Start>{str(coord[0])} & End<{str(coord[1])}'), ax[4], '#855B54', 'ABC Powerlaw', score='ABC Powerlaw' )
    plot_bedgraph(atac_bw.df.query(f'Chromosome=="{chrom}" & Start>{str(coord[0])} & End<{str(coord[1])}').rename(columns={'HUDEP ATAC':'atac'}), ax[5], '#b3b3b3', label= "HUDEP-2", score='atac', dot=False)
    plot_bedgraph(annot_hg38.query(f'Chromosome=="{chrom}" & Start>{str(coord[0])} & End<{str(coord[1])}'), ax[6], '#D483B8', label= 'Fetal\nErythroblast', score='ATAC', absolute=False, dot=False)
    plot_bedpe(filter_hichip_by_tss(chrom, tss_start, hichip).query(f'Chromosome1=="{chrom}" & Start1>{str(coord[0])} & End2<{str(coord[1])}'), coord[0], coord[1] , ax[7], '#b3b3b3', 'HiChIP')

    # plot_gene_tss(cell, gene, ax[2], 'red', gene, score='exp', bar_length = 10)
    for i, axis in enumerate(ax):
        axis.set_xlim(coord[0], coord[1])
        if i != 7:  
            axis.set_ylim(0, 1.1)
        axis.legend().remove()
        axis.set_xticks([])
        axis.set_yticks([])
        # keep only bottom and left spines
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    

    # set xticks unit to Mbp, keep 2 decimal places
    ax[1].xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(float(x/1000000), '.2f')))
    # remove ax[3] ticks
    
    ax[7].set_ylabel('HUDEP-2', rotation=0, ha='right', va='center')
    # 2 decimal in title
    ax[0].set_title(f'Chr{chrom[3:]}:{coord[0]}-{coord[1]} ({(coord[1]-(coord[0]))/1000000:.2f}Mbp)')
    print(f'Chr{chrom[3:]}:{coord[0]}-{coord[1]}')
    return fig, ax
# %%
g = 'MYB'
# 134980972-135773328
fig, ax = plot_gene_crispr(g, final_result_hg38_coord, (134980972,135773328))
 # avoid y ticks being cropped off, add padding
plt.tight_layout(h_pad=1)
plt.savefig(f'track_{g}.png', dpi=300)

# %%
g = 'BCL11A'
# 60424394-60574394
fig, ax = plot_gene_crispr(g, final_result_hg38_coord, (60324394,61074394))
plt.tight_layout()
plt.savefig(f'track_{g}.png', dpi=300)
#%%
g = 'NFIX'
# 12694852-13794852
fig, ax = plot_gene_crispr(g, final_result_hg38_coord, (12694852,13794852))
plt.tight_layout()
plt.savefig(f'track_{g}.png', dpi=300)
#%%
g = 'HBG2'
# 4276011-6276011
fig, ax = plot_gene_crispr(g, final_result_hg38_coord, (4276011,6276011))
plt.tight_layout()
plt.savefig(f'track_{g}.png', dpi=300)

# %%
