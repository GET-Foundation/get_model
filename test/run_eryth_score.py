#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from atac_rna_data_processing.config.load_config import load_config as load_data_config
from atac_rna_data_processing.io.celltype import *
from atac_rna_data_processing.io.mutation import *
from atac_rna_data_processing.io.nr_motif_v1 import *
from atac_rna_data_processing.io.region import *
from pyranges import PyRanges as prs, read_bed
import torch
from torch import nn
import s3fs
from tqdm import tqdm

from get_model.run_region import RegionLitModel
#%%
from get_model.utils import load_config
cfg = load_config('fetal_region_v2')
model = RegionLitModel(cfg).model 
#%%
# Load configuration
GET_CONFIG = load_data_config('/home/xf2217/Projects/atac_rna_data_processing/atac_rna_data_processing/config/GET')
GET_CONFIG.celltype.jacob = True
GET_CONFIG.celltype.embed = False
GET_CONFIG.celltype.num_cls = 2
GET_CONFIG.assets_dir = 'assets'

s3_uri = "s3://2023-get-xf2217/get_demo_test_data"
s3_file_sys = s3fs.S3FileSystem(anon=True)
GET_CONFIG.s3_file_sys = s3_file_sys
GET_CONFIG.celltype.data_dir = f"{s3_uri}/pretrain_human_bingren_shendure_apr2023/fetal_adult/"
GET_CONFIG.celltype.interpret_dir = f"{s3_uri}/Interpretation_all_hg38_allembed_v4_natac/"
GET_CONFIG.motif_dir = f"{s3_uri}/interpret_natac/motif-clustering/"
GET_CONFIG.assets_dir = f"{s3_uri}/assets/"

# Load cell type data
cell = GETCellType('118', GET_CONFIG)

#%%
# Load model
class WrapperModel(nn.Module):
    def __init__(self, m, focus):
        super().__init__()
        self.model = m
        self.focus = focus
    
    def forward(self, x, strand):
        return self.model(x)[:, self.focus, strand]

def get_gene_tad_idx(gene, cell, tad, num_shift=3):
    if tad is None:
        return get_gene_idx(gene, cell, focus=100)
    tss_list = cell.get_gene_tss(gene)
    idx_list = []
    for tss in tss_list:
        for i in range(num_shift):
            shift = np.random.randint(-10, 10)
            gene_tad = tad[(tad.Chromosome == tss.chrom) & (tad.Start <= tss.start) & (tad.End >= tss.start)]
            idx = gene_tad.join(prs(cell.peak_annot)).df['index'].values
            tad_left, tad_right = idx[0], idx[-1]
            init_start_idx = tss.peak_id - 100 + shift
            init_end_idx = tss.peak_id + 100 + shift
            left_boundary = max(init_start_idx, tad_left)
            right_boundary = min(init_end_idx, tad_right)
            idx_list.append((left_boundary, right_boundary, tss.peak_id - left_boundary, right_boundary - left_boundary))
    return idx_list

def prepare_input(cell, start_idx, end_idx, use_natac=True):
    input_data = cell.get_input_data(start=start_idx, end=end_idx)
    atac = input_data[:, 282]
    if use_natac:
        input_data[:, 282] = 1
    input_data = torch.FloatTensor(input_data).cuda().unsqueeze(0)
    if input_data.shape[1] < 200:
        input_data = torch.cat([input_data, torch.zeros(input_data.shape[0], 200-input_data.shape[1], input_data.shape[2]).cuda()], 1)
    elif input_data.shape[1] >= 200:
        input_data = input_data[:, :200, :]
    return input_data, atac

def minmax(x):
    return (x - x.min())/(x.max()-x.min())

def softmax(x):
    return np.exp(x)/np.exp(x).sum()

def interpret_step(model, input_data, focus, strand):
    target_tensors = {}
    hooks = []
    
    def capture_target_tensor(name):
        def hook(module, input, output):
            output.retain_grad()
            target_tensors[name] = output
        return hook
    
    layer = model.region_embed
    hook = layer.register_forward_hook(capture_target_tensor('region_embed'))
    hooks.append(hook)
    input_data.requires_grad = True
    output = model(input_data)
    pred = output[:, focus, strand]
    pred.backward()
    
    jacobian = input_data.grad.detach().cpu().float().numpy().squeeze(0)
    jacobian_norm = np.linalg.norm(jacobian, axis=1)
    # jacobian_norm = jacobian*input_data.detach().cpu().float().numpy().squeeze(0)
    
    for hook in hooks:
        hook.remove()
    
    return jacobian_norm

def get_jacobian_prediction(gene, cell, model, tad, use_natac=True):
    tss = cell.get_gene_tss(gene)[0]
    idx_list = get_gene_tad_idx(gene, cell, tad)
    print(idx_list)
    peak_annot_list = []
    for start_idx, end_idx, focus, num_region in idx_list:
        input_data, atac = prepare_input(cell, start_idx, end_idx, use_natac=use_natac)
        jacobian_norm = interpret_step(model, input_data, focus, tss.strand)
        peak_annot = cell.peak_annot[start_idx:end_idx].copy().drop('index', axis=1)
        peak_annot['jacobian_norm'] = jacobian_norm[:num_region]
        peak_annot['atac'] = atac
        peak_annot_list.append(peak_annot)
    
    peak_annot = pd.concat(peak_annot_list, axis=0)
    peak_annot = peak_annot.groupby(['Chromosome', 'Start', 'End']).mean().reset_index()
    return peak_annot

def get_gene_idx(gene, cell, focus=100):
    tss_list = cell.get_gene_tss(gene)
    idx_list = []
    for tss in tss_list:
        start_idx, end_idx = tss.peak_id - focus, tss.peak_id - focus + 200
        num_region = end_idx - start_idx
        idx_list.append((start_idx, end_idx, focus, num_region))
    return idx_list

def run_interp(gene, cell, tad, use_natac=True, shift=3):
    tss = cell.get_gene_tss(gene)[0]
    idx_list = get_gene_tad_idx(gene, cell, tad, shift)
    
    jacobian_result = get_jacobian_prediction(gene, cell, model, tad, use_natac=use_natac)
    
    return {'jacobian_norm': jacobian_result, 'atac': jacobian_result}

def run_interp_in_tad(gene, cell, tad, ground_truth, use_natac=True):
    chrom = cell.get_gene_tss(gene)[0].chrom
    result_dict = run_interp(gene, cell, tad, use_natac, shift=3)
    if isinstance(ground_truth, prs):
        ground_truth = ground_truth.df
    
    final_df = result_dict['jacobian_norm'].iloc[:, 0:3].copy()
    for attr_name in result_dict:
        final_df[attr_name] = result_dict[attr_name][attr_name]
    
    return final_df

def calc_gene_crispr(cell, gene, tad, locus):
    final_df = run_interp_in_tad(gene, cell, None, locus, use_natac=False)
    return final_df
#%%
# Main execution
model.cuda()
tad = read_bed('/home/xf2217/Projects/geneformer_nova/data/tad.hg38.sort_uniq.bed')
locus = prs(read_bed("/home/xf2217/Projects/geneformer_esc/eryth/locus.bed", as_df=True).reset_index(), int64=True)

final_overlap_df_list = []
for gene in ['MYB', 'NFIX', 'BCL11A', 'HBG2']:
    final_overlap_df = calc_gene_crispr(cell, gene, tad, locus)
    final_overlap_df['Gene'] = gene
    final_overlap_df_list.append(final_overlap_df)

final_overlap_df_final = pd.concat(final_overlap_df_list)
final_overlap_df_final.to_csv('eryth_final.csv')
# %%







