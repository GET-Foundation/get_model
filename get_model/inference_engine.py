# inference_engine.py
import numpy as np
import torch
from pyranges import PyRanges as pr
from get_model.dataset.collate import get_rev_collate_fn
from get_model.dataset.zarr_dataset import PreloadDataPack, get_padding_pos
from get_model.model.model import GETFinetuneExpATAC, GETFinetuneExpATACFromSequence
from get_model.engine import train_class_batch

class ModelWrapper:
    def __init__(self, checkpoint_path, device='cuda', with_sequence=False, model=None):
        if model is not None:
            self.model = model
        else:
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
        sample_track, peak_seq, sample_metadata, celltype_peaks, sample_track_boundary, sample_peak_sequence_boundary, chunk_size, mask, n_peaks, max_n_peaks, total_peak_len, motif_mean_std, labels_data, other_labels_data, hic_matrix  = batch_data
        sample_track = sample_track.to(device, non_blocking=True).float()
        peak_seq = peak_seq.to(device, non_blocking=True).float()
        motif_mean_std = motif_mean_std.to(device, non_blocking=True).float()
        n_peaks = n_peaks.to(device, non_blocking=True)
        labels_data = labels_data.to(device, non_blocking=True).float()
        other_labels_data = other_labels_data.to(device, non_blocking=True).float()
        hic_matrix = hic_matrix.to(device, non_blocking=True).float()
        
        # compute output
        loss, exp, exp_target, atac, atac_target, confidence_pred, confidence_target = train_class_batch(self.model, peak_seq, sample_track, mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, other_labels_data[:,:,0], labels_data, other_labels_data, self.loss, hic_matrix)

        # other_labels_data is B, R, N where [:,:, 1] is TSS indicator
        # only append tss preds and obs
        pred_exp = exp.reshape(-1).detach().cpu().numpy()
        ob_exp = exp_target.reshape(-1).detach().cpu().numpy()
        pred_atac = atac.reshape(-1).detach().cpu().numpy()
        ob_atac = other_labels_data[:,:,0].float().reshape(-1).detach().cpu().numpy()
        return pred_exp, ob_exp, pred_atac, ob_atac


        
class InferenceEngine:
    def __init__(self, dataset, model_checkpoint, mut=None, peak_inactivation=None, with_sequence=False, device='cuda', model=None):
        self.dataset = dataset
        self.model_wrapper = ModelWrapper(model_checkpoint, with_sequence=with_sequence, device=device, model=model)
        self.mut = mut
        self.peak_inactivation = peak_inactivation

    def setup_data(self, gene_name, celltype, window_idx_offset=0, track_start=None, track_end=None):
        data_key = self.dataset.datapool.data_keys[0]
        dataset =  self.dataset
        window_idx = self.dataset._get_window_idx_for_gene_and_celltype(data_key, celltype, gene_name)['window_idx']
        self.window_idx = window_idx
        self.celltype = celltype
        self.gene_name = gene_name
        self.gene_info = self.dataset._get_gene_info_from_window_idx(window_idx[0]+window_idx_offset).query('gene_name==@gene_name')
        self.chr_name = self.gene_info['Chromosome'].values[0]

        peak_info = self.dataset.datapool.zarr_dict[data_key].get_peaks(celltype, 'peaks_q0.01_tissue_open_exp').reset_index(drop=True).reset_index()
        self.data_key = data_key
        self.peak_info = peak_info
        track_start, track_end, tss_peak, peak_start, strand = self.get_peak_start_end_from_gene_peak(self.gene_info, self.peak_info, gene_name, track_start, track_end)
        self.peak_start = peak_start
        self.strand = strand
        self.track_start = track_start
        self.track_end = track_end
        self.tss_peak = np.unique(tss_peak)
        self.data = self.dataset.datapool.load_data(
            data_key=self.data_key, 
            celltype_id=self.celltype,
            chr_name=self.chr_name,
            start=self.track_start,
            end=self.track_end)

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

        batch = self.dataset.datapool.generate_sample(chr_name, start, end, self.data_key, celltype_id, track, celltype_peaks, motif_mean_std, mut=mut, peak_inactivation=peak_inactivation)
        tss_peak = self.tss_peak - offset
        mut_peak = None
        if mut is not None:
            mut_peak = pr(self.peak_info).join(pr(mut)).df['index'].values - offset - self.peak_start

        # 7. Run the model inference
        inference_results, prepared_batch = self.run_inference_with_batch(batch)

        # 8. Handle the inference results as needed
        return inference_results, prepared_batch, tss_peak, mut_peak
    
    def run_inference_with_batch(self, batch):
        prepared_batch = get_rev_collate_fn([batch])
        inference_results = self.model_wrapper.infer(prepared_batch)
        inference_results = {'pred_exp': inference_results[0], 'ob_exp': inference_results[1], 'pred_atac': inference_results[2], 'ob_atac': inference_results[3]}
        # peak_signal_track, peak_sequence, sample_metadata, celltype_peaks, peak_signal_track_boundary, peak_sequence_boundary, chunk_size, mask, n_peaks, max_n_peaks, total_peak_len, motif_mean_std, exp_label, other_peak_labels = prepared_batch
        prepared_batch = {'peak_signal_track': prepared_batch[0], 'peak_sequence': prepared_batch[1], 'sample_metadata': prepared_batch[2], 'celltype_peaks': prepared_batch[3], 'peak_signal_track_boundary': prepared_batch[4], 'peak_sequence_boundary': prepared_batch[5], 'chunk_size': prepared_batch[6], 'mask': prepared_batch[7], 'n_peaks': prepared_batch[8], 'max_n_peaks': prepared_batch[9], 'total_peak_len': prepared_batch[10], 'motif_mean_std': prepared_batch[11], 'exp_label': prepared_batch[12], 'other_peak_labels': prepared_batch[13]}
        return inference_results, prepared_batch
    
    def get_peak_start_end_from_gene_peak(self, gene_info, peak_info, gene, track_start=None, track_end=None):
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
        strand = gene_df.Strand.replace({'+': 1, '-': -1}).values[0]
        if gene_df.shape[0] == 0:
            raise ValueError(f"Gene {gene} not found in the peak information.")
        idx = gene_df.aTPM.idxmax()
        if track_start is None or track_end is None:
            # Get the peak start and end positions based on n_peaks_upper_bound
            peak_start, peak_end = df.iloc[idx]['index'] - self.dataset.n_peaks_upper_bound // 2, df.iloc[idx]['index'] + self.dataset.n_peaks_upper_bound // 2
            tss_peak = gene_df['index'].values
            tss_peak = tss_peak - peak_start
        else:
            peak_info_subset = peak_info.query('Chromosome==@self.chr_name').query('Start>=@track_start & End<=@track_end')
            peak_start, peak_end = peak_info_subset.index.min(), peak_info_subset.index.max()
            tss_peak = pr(peak_info_subset).join(pr(gene_df)).df['index'].values
            tss_peak = tss_peak - peak_start
        track_start = peak_info.iloc[peak_start].Start
        track_end = peak_info.iloc[peak_end].End

        return track_start, track_end, tss_peak, peak_start, strand