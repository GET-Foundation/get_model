import sys
import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyranges import PyRanges as pr
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool


# plt.style.use('manuscript.mplstyle')
sys.path.append('/pmglocal/alb2281/repos/atac_rna_data_processing')
sys.path.append('/manitou/pmg/users/xf2217/get_model/')

from atac_rna_data_processing.config.load_config import load_config
from atac_rna_data_processing.io.mutation import read_rsid_parallel, Mutations, MutationsInCellType
from atac_rna_data_processing.io.nr_motif_v1 import NrMotifV1
from atac_rna_data_processing.io.celltype import GETCellType
from atac_rna_data_processing.io.causal_lib import *
from atac_rna_data_processing.io.region import *
from inference import InferenceModel


GET_CONFIG = load_config('/manitou/pmg/users/xf2217/atac_rna_data_processing/atac_rna_data_processing/config/GET')
GET_CONFIG.celltype.jacob=False
GET_CONFIG.celltype.num_cls=2
GET_CONFIG.celltype.input=True
GET_CONFIG.celltype.embed=False
GET_CONFIG.assets_dir=''
GET_CONFIG.s3_file_sys=''
GET_CONFIG.celltype.data_dir = '/manitou/pmg/users/xf2217/pretrain_human_bingren_shendure_apr2023/fetal_adult/'
GET_CONFIG.celltype.interpret_dir='/manitou/pmg/users/xf2217/Interpretation_all_hg38_allembed_v4_natac/'
cell_type_annot = pd.read_csv("/manitou/pmg/users/xf2217/pretrain_human_bingren_shendure_apr2023/data/cell_type_pretrain_human_bingren_shendure_apr2023.txt")
cell_type_annot_dict = cell_type_annot.set_index('id').celltype.to_dict()

checkpoint_path = '/manitou/pmg/projects/resources/get_interpret/pretrain_finetune_natac_fetal_adult.pth'
working_dir = "/manitou/pmg/users/xf2217/interpret_natac/"
inf_model = InferenceModel(checkpoint_path, 'cuda')
num_workers = 10


hg38 = Genome('hg38', working_dir + "/hg38.fa")
motif = NrMotifV1.load_from_pickle(working_dir + "/NrMotifV1.pkl")
variants_rsid = read_rsid_parallel(hg38, working_dir + 'myc_rsid.txt', 5)

normal_variants = pd.read_csv('/manitou/pmg/users/xf2217/gnomad/myc.tad.vcf.gz', sep='\t', comment='#', header=None)
normal_variants.columns = ['Chromosome', 'Start', 'RSID', 'Ref', 'Alt', 'Qual', 'Filter', 'Info']
normal_variants['End'] = normal_variants.Start
normal_variants['Start'] = normal_variants.Start-1
normal_variants = normal_variants[['Chromosome', 'Start', 'End', 'RSID', 'Ref', 'Alt', 'Qual', 'Filter', 'Info']]
normal_variants = normal_variants.query('Ref.str.len()==1 & Alt.str.len()==1')
normal_variants['AF'] = normal_variants.Info.transform(lambda x: float(re.findall(r'AF=([0-9e\-\.]+)', x)[0]))
normal_variants_df = normal_variants.copy().query('AF>0.01').drop_duplicates(subset='RSID').query('RSID!="." & RSID!="rs55705857"')

breakpoint()

CellCollection = {}
CellMutCollection = {}

def predict_expression_per_cell_type(cell_id):
    results = []
    cell_id = os.path.basename(cell_id)
    cell_type = cell_type_annot_dict[cell_id]
    CellCollection[cell_type] = GETCellType(cell_id, GET_CONFIG)
    if pr(CellCollection[cell_type].peak_annot).join(pr(variants_rsid.df)).df.empty:
        results.append([cell_type, 1])
        return 
    
    cell_mut = MutationsInCellType(hg38, variants_rsid.df, CellCollection[cell_type])
    cell_mut.get_original_input(motif)
    cell_mut.get_altered_input(motif)
    CellMutCollection[cell_type] = cell_mut
    ref_exp, alt_exp = cell_mut.predict_expression('rs55705857', 'MYC', 100, 200, inf_model=inf_model)
    return [cell_type, alt_exp/ref_exp]

cell_types = sorted(glob('/manitou/pmg/users/xf2217/Interpretation_all_hg38_allembed_v4_natac/*'))
# with Pool(processes=num_workers) as p:
#     mp_result_col = p.map(
#         predict_expression_per_cell_type, tqdm(cell_types, total=len(cell_types)),
#     )

# debug in series
for cell_id in tqdm(cell_types):
    predict_expression_per_cell_type(cell_id)
