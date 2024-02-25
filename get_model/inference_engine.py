# inference_engine.py
import numpy as np
import torch

from get_model.dataset.collate import get_rev_collate_fn
from get_model.dataset.zarr_dataset import PreloadDataPack, get_padding_pos
from get_model.model.model import GETFinetuneExpATAC, GETFinetuneExpATACFromSequence


class ModelWrapper:
    def __init__(self, checkpoint_path, device='cuda', with_sequence=False):
        if with_sequence:
            self.model = GETFinetuneExpATACFromSequence(
                num_regions=100,
                num_res_block=0,
                motif_prior=False,
                embed_dim=768,
                num_layers=12,
                d_model=768,
                flash_attn=False,
                nhead=12,
                dropout=0.1,
                output_dim=2,
                pos_emb_components=[],
            )
        else:
            self.model = GETFinetuneExpATAC(
                num_regions=100,
                num_res_block=0,
                motif_prior=False,
                embed_dim=768,
                num_layers=12,
                d_model=768,
                flash_attn=False,
                nhead=12,
                dropout=0.1,
                output_dim=2,
                pos_emb_components=[],
                atac_kernel_num=161,
                atac_kernel_size=3,
                joint_kernel_num=161,
                final_bn=False,
            )
        self.device=device
        self._load_checkpoint(checkpoint_path)
        self.loss = torch.nn.PoissonNLLLoss(reduction='mean', log_input=False)
        
    def _load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"], strict=False)
        self.model.eval()
        self.model.to(self.device)

    def infer(self, batch_data):
        device = self.device
        # Placeholder for inference code
        sample_track, peak_seq, sample_metadata, celltype_peaks, sample_track_boundary, sample_peak_sequence_boundary, chunk_size, mask, n_peaks, max_n_peaks, total_peak_len, motif_mean_std, labels_data, other_labels_data  = batch_data
        sample_track = sample_track.to(device, non_blocking=True).float()
        peak_seq = peak_seq.to(device, non_blocking=True).float()
        motif_mean_std = motif_mean_std.to(device, non_blocking=True).float()
        n_peaks = n_peaks.to(device, non_blocking=True)
        labels_data = labels_data.to(device, non_blocking=True).float()
        other_labels_data = other_labels_data.to(device, non_blocking=True).float()
        
        # compute output
        loss, atac, exp, exp_target = train_class_batch(self.model, peak_seq, sample_track, mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, other_labels_data[:,:,0], labels_data, self.loss, other_labels_data)

        # other_labels_data is B, R, N where [:,:, 1] is TSS indicator
        # only append tss preds and obs
        pred_exp = exp.reshape(-1).detach().cpu().numpy()
        ob_exp = exp_target.reshape(-1).detach().cpu().numpy()
        pred_atac = atac.reshape(-1).detach().cpu().numpy()
        ob_atac = other_labels_data[:,:,0].float().reshape(-1).detach().cpu().numpy()
        return pred_exp, ob_exp, pred_atac, ob_atac


def train_class_batch(model, peak_seq, sample_track, mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, atac_target, exp_target, criterion,other_labels_data):
    device = peak_seq.device
    padding_mask = get_padding_pos(mask)
    mask_for_loss = 1-padding_mask
    padding_mask = padding_mask.to(device, non_blocking=True).bool()
    mask_for_loss = mask_for_loss.to(device, non_blocking=True).unsqueeze(-1)
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        atac, exp, confidence = model(peak_seq, sample_track, mask_for_loss, padding_mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, other_labels_data)
        

    exp = exp * mask_for_loss
    indices = torch.where(mask_for_loss==1)
    exp = exp[indices[0], indices[1], :].flatten()
    exp_target = exp_target * mask_for_loss
    exp_target = exp_target[indices[0], indices[1], :].flatten()
    loss_exp = criterion(exp, exp_target)
    
    if atac is not None:
        atac = atac * mask_for_loss
        indices = torch.where(mask_for_loss==1)
        atac = atac[indices[0], indices[1], :].flatten()
        atac_target = atac_target.unsqueeze(-1) * mask_for_loss
        atac_target = atac_target[indices[0], indices[1], :].flatten()
        loss_atac = criterion(atac, atac_target)
        loss = loss_exp + loss_atac 
    else:
        loss = loss_exp
    return loss, atac, exp, exp_target 

class InferenceEngine:
    def __init__(self, dataset, model_checkpoint, mut=None, peak_inactivation=None, with_sequence=False):
        self.dataset = dataset
        self.model_wrapper = ModelWrapper(model_checkpoint, with_sequence=with_sequence)
        self.mut = mut
        self.peak_inactivation = peak_inactivation

    def setup_data(self, gene_name, celltype, window_idx_offset=0):
        data_key = self.dataset.datapool.data_keys[0]
        dataset =  self.dataset
        window_idx = self.dataset._get_window_idx_for_gene_and_celltype(data_key, celltype, gene_name)['window_idx']
        self.window_idx = window_idx
        self.celltype = celltype
        self.gene_name = gene_name
        self.gene_info = self.dataset._get_gene_info_from_window_idx(window_idx[0]+window_idx_offset).query('gene_name==@gene_name')
        self.chr_name = self.gene_info['Chromosome'].values[0]

        peak_info = self.dataset.datapool.zarr_dict[data_key].get_peaks(celltype, 'peaks_q0.01_tissue_open_exp')
        self.data_key = data_key
        self.peak_info = peak_info
        peak_start, peak_end, tss_peak = self.get_peak_start_end_from_gene_peak(self.gene_info, self.peak_info, gene_name)
        self.peak_start = peak_start
        self.peak_end = peak_end
        self.tss_peak = np.unique(tss_peak)
        self.data = self.dataset.datapool.load_data(
            data_key=self.data_key, 
            celltype_id=self.celltype,
            chr_name=self.chr_name,
            start=self.peak_start,
            end=self.peak_end)

    def run_inference_for_gene_and_celltype(self, offset=0, mut=None, peak_inactivation=None):
        """
        Run inference for a specific gene and cell type.

        Parameters:
        - gene_name (str): The name of the gene.
        - celltype (str): The cell type.
        """
        if mut is None:
            mut = self.mut
        if peak_inactivation is None:
            peak_inactivation = self.peak_inactivation
            
        # 1. Get the gene and cell type information
        chr_name, start, end, celltype_id, track, item_insulation, celltype_peaks, motif_mean_std = self.data
        # 5. Extract the sample data for inference

        batch = self.dataset.datapool.generate_sample(chr_name, start, end, celltype_id, track, celltype_peaks, motif_mean_std, mut=None, peak_inactivation=None)
        tss_peak = self.tss_peak - offset

        # 6. Prepare the batch data for the model (assuming a method exists for this)
        prepared_batch = get_rev_collate_fn([batch])

        # 7. Run the model inference
        inference_results = self.model_wrapper.infer(prepared_batch)

        # 8. Handle the inference results as needed
        return inference_results, prepared_batch, tss_peak
    
    def get_peak_start_end_from_gene_peak(self, gene_info, peak_info, gene):
        """
        Determine the peak start and end positions for a specific gene.

        Parameters:
        - gene_info (DataFrame): Gene information DataFrame.
        - peak_info (DataFrame): Peak information DataFrame.
        - gene (str): The name of the gene.

        Returns:
        - tuple: The start and end indices of the peak.
        """
        # Assuming PyRanges is used for managing genomic ranges
        from pyranges import PyRanges as pr
        df = pr(peak_info.copy().reset_index(drop=True).reset_index()).join(pr(gene_info.copy().query('gene_name==@gene')[['Chromosome', 'Start', 'End', 'gene_name', 'Strand', 'chunk_idx']].drop_duplicates()).extend(300), suffix="_gene", how='left', apply_strand_suffix=False).df[['index', 'Chromosome', 'Start', 'End', 'Expression_positive', 'Expression_negative', 'aTPM', 'TSS', 'gene_name', 'Strand', 'chunk_idx']].reset_index(drop=True)
        gene_df = df.query('gene_name==@gene')
        if gene_df.shape[0] == 0:
            raise ValueError(f"Gene {gene} not found in the peak information.")
        idx = gene_df.aTPM.idxmax()
        peak_start, peak_end = df.iloc[idx]['index'] - self.dataset.n_peaks_upper_bound // 2, df.iloc[idx]['index'] + self.dataset.n_peaks_upper_bound // 2
        print(peak_start, peak_end)
        peak_start = peak_info.iloc[peak_start].Start
        peak_end = peak_info.iloc[peak_end].End
        # peak_start = max(0, peak_start)
        # peak_end = min(peak_end, peak_info['index'].max())
        tss_peak = gene_df['index'].values
        tss_peak = tss_peak - peak_start
        return peak_start, peak_end, tss_peak