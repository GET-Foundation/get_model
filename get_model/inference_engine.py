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
        self.device = device
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
        sample_track, peak_seq, sample_metadata, celltype_peaks, sample_track_boundary, sample_peak_sequence_boundary, chunk_size, mask, n_peaks, max_n_peaks, total_peak_len, motif_mean_std, labels_data, other_labels_data, hic_matrix = batch_data
        sample_track = sample_track.to(device, non_blocking=True).float()
        peak_seq = peak_seq.to(device, non_blocking=True).float()
        motif_mean_std = motif_mean_std.to(device, non_blocking=True).float()
        n_peaks = n_peaks.to(device, non_blocking=True)
        labels_data = labels_data.to(device, non_blocking=True).float()
        other_labels_data = other_labels_data.to(
            device, non_blocking=True).float()
        hic_matrix = hic_matrix.to(device, non_blocking=True).float()

        # compute output
        results = train_class_batch(self.model, peak_seq, sample_track, mask, chunk_size, n_peaks, max_n_peaks,
                                    motif_mean_std, other_labels_data, labels_data, other_labels_data, self.loss, hic_matrix)

        # other_labels_data is B, R, N where [:,:, 1] is TSS indicator
        # only append tss preds and obs
        if self.model._get_name() == 'GETFinetuneChrombpNet':
            atpm_pred = results['atpm_pred']
            atpm_target = results['atpm_target']
            aprofile_pred = results['aprofile_pred']
            aprofile_target = results['aprofile_target']
            return atpm_pred, atpm_target, aprofile_pred, aprofile_target
        else:
            pred_exp = exp.reshape(-1).detach().cpu().numpy()
            ob_exp = exp_target.reshape(-1).detach().cpu().numpy()
            pred_atac = atac.reshape(-1).detach().cpu().numpy()
            ob_atac = other_labels_data[:, :, 0].float(
            ).reshape(-1).detach().cpu().numpy()
            return pred_exp, ob_exp, pred_atac, ob_atac


class InferenceEngine:
    def __init__(self, dataset, model_checkpoint, mut=None, peak_inactivation=None, with_sequence=False, device='cuda', model=None):
        self.dataset = dataset
        self.model_wrapper = ModelWrapper(
            model_checkpoint, with_sequence=with_sequence, device=device, model=model)
        self.mut = mut
        self.peak_inactivation = peak_inactivation

    def setup_data(self, gene_name, celltype, window_idx_offset=0, track_start=None, track_end=None):
        data_key = self.dataset.datapool.data_keys[0]
        dataset = self.dataset
        window_idx = self.dataset._get_window_idx_for_gene_and_celltype(
            data_key, celltype, gene_name)['window_idx']
        self.window_idx = window_idx
        self.celltype = celltype
        self.gene_name = gene_name
        self.gene_info = self.dataset._get_gene_info_from_window_idx(
            window_idx[0]+window_idx_offset).query('gene_name==@gene_name')
        self.chr_name = self.gene_info['Chromosome'].values[0]
        self.tss_coord = self.gene_info.Start.values[0]

        peak_info = self.dataset.datapool._query_peaks(
            celltype, self.chr_name, self.tss_coord-4000000, self.tss_coord+4000000).reset_index(drop=True).reset_index()
        self.data_key = data_key
        self.peak_info = peak_info
        track_start, track_end, tss_peak, peak_start, strand = self.get_peak_start_end_from_gene_peak(
            self.gene_info, self.peak_info, gene_name, track_start, track_end)
        self.peak_start = peak_start
        self.strand = strand
        self.track_start = track_start
        self.track_end = track_end
        self.tss_peak = np.unique(tss_peak)

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

        batch = self.dataset.datapool.generate_sample(
            self.chr_name, self.track_start, self.track_end, self.data_key, self.celltype, mut=mut, peak_inactivation=peak_inactivation)
        tss_peak = self.tss_peak - offset
        mut_peak = None
        if mut is not None:
            mut_peak = pr(self.peak_info).join(
                pr(mut)).df['index'].values - offset - self.peak_start

        # 7. Run the model inference
        inference_results, prepared_batch = self.run_inference_with_batch(
            batch)

        # 8. Handle the inference results as needed
        return inference_results, prepared_batch, tss_peak, mut_peak

    def run_inference_with_batch(self, batch):
        prepared_batch = get_rev_collate_fn([batch])
        inference_results = self.model_wrapper.infer(prepared_batch)
        inference_results = {'pred_exp': inference_results[0], 'ob_exp': inference_results[1],
                             'pred_atac': inference_results[2], 'ob_atac': inference_results[3]}
        # peak_signal_track, peak_sequence, sample_metadata, celltype_peaks, peak_signal_track_boundary, peak_sequence_boundary, chunk_size, mask, n_peaks, max_n_peaks, total_peak_len, motif_mean_std, exp_label, other_peak_labels = prepared_batch
        prepared_batch = {'peak_signal_track': prepared_batch[0], 'peak_sequence': prepared_batch[1], 'sample_metadata': prepared_batch[2], 'celltype_peaks': prepared_batch[3], 'peak_signal_track_boundary': prepared_batch[4], 'peak_sequence_boundary': prepared_batch[5],
                          'chunk_size': prepared_batch[6], 'mask': prepared_batch[7], 'n_peaks': prepared_batch[8], 'max_n_peaks': prepared_batch[9], 'total_peak_len': prepared_batch[10], 'motif_mean_std': prepared_batch[11], 'exp_label': prepared_batch[12], 'other_peak_labels': prepared_batch[13]}
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
        columns_to_include = ['index', 'Chromosome', 'Start',
                              'End',  'gene_name', 'Strand', 'chunk_idx']
        if self.dataset.additional_peak_columns is not None:
            columns_to_include += self.dataset.additional_peak_columns
        df = pr(peak_info.copy().reset_index()).join(pr(gene_info.copy().query('gene_name==@gene')[['Chromosome', 'Start', 'End', 'gene_name', 'Strand', 'chunk_idx']].drop_duplicates(
        )).extend(300), suffix="_gene", how='left', apply_strand_suffix=False).df[columns_to_include].set_index('index')
        gene_df = df.query('gene_name==@gene')
        strand = gene_df.Strand.replace({'+': 1, '-': -1}).values[0]
        if gene_df.shape[0] == 0:
            raise ValueError(f"Gene {gene} not found in the peak information.")
        # tss_peak = gene_df.aTPM.idxmax()
        tss_peak = gene_df.index.values[0]
        if track_start is None or track_end is None:
            # Get the peak start and end positions based on n_peaks_upper_bound
            peak_start, peak_end = tss_peak - self.dataset.n_peaks_upper_bound // 2, tss_peak + \
                self.dataset.n_peaks_upper_bound // 2
            tss_peak = tss_peak - peak_start
        else:
            peak_info_subset = peak_info.query(
                'Chromosome==@self.chr_name').query('Start>=@track_start & End<=@track_end')
            peak_start, peak_end = peak_info_subset.index.min(), peak_info_subset.index.max()
            tss_peak = pr(peak_info_subset).join(
                pr(gene_df)).df['index'].values
            tss_peak = tss_peak - peak_start
        track_start = peak_info.iloc[peak_start].Start-self.dataset.padding
        track_end = peak_info.iloc[peak_end].End+self.dataset.padding

        return track_start, track_end, tss_peak, peak_start, strand


class InferenceEngine:
    def __init__(self, dataset, model_checkpoint, with_sequence=False, device='cuda', model=None):
        self.dataset = dataset
        self.model_wrapper = ModelWrapper(
            model_checkpoint, with_sequence=with_sequence, device=device, model=model)

    def run_inference_with_batch(self, sample):
        batch = [sample]
        prepared_batch = get_rev_collate_fn(batch)
        inference_results = self.model_wrapper.infer(prepared_batch)
        inference_results = {
            'pred_exp': inference_results['pred_exp'],
            'ob_exp': inference_results['ob_exp'],
            'pred_atac': inference_results['pred_atac'],
            'ob_atac': inference_results['ob_atac']
        }

        prepared_batch = {
            'sample_track': prepared_batch['sample_track'],
            'sample_peak_sequence': prepared_batch['sample_peak_sequence'],
            'metadata': prepared_batch['metadata'],
            'celltype_peaks': prepared_batch['celltype_peaks'],
            'chunk_size': prepared_batch['chunk_size'],
            'mask': prepared_batch['mask'],
            'n_peaks': prepared_batch['n_peaks'],
            'max_n_peaks': prepared_batch['max_n_peaks'],
            'total_peak_len': prepared_batch['total_peak_len'],
            'motif_mean_std': prepared_batch['motif_mean_std'],
            'exp_label': prepared_batch['exp_label'],
            'atpm': prepared_batch['atpm'],
            'hic_matrix': prepared_batch['hic_matrix']
        }
        return inference_results, prepared_batch

    def run_inference_for_gene_and_celltype(self, gene_name, celltype, track_start=None, track_end=None, offset=0, mut=None, peak_inactivation=None):
        item = self.dataset.get_item_for_gene_in_celltype(
            gene_name, celltype, track_start, track_end, offset, mut, peak_inactivation)
        sample = item['sample']
        tss_peak = item['tss_peak']
        mut_peak = item['mut_peak']

        inference_results, prepared_batch = self.run_inference_with_batch(
            sample)

        return inference_results, prepared_batch, tss_peak, mut_peak


class VariantInferenceEngine(InferenceEngine):
    def __init__(self, dataset, model_checkpoint, mut, with_sequence=False, device='cuda', model=None):
        super().__init__(dataset, model_checkpoint, mut=mut,
                         with_sequence=with_sequence, device=device, model=model)

    def setup_data(self, variant_info, celltype, window_idx_offset=0):
        data_key = self.dataset.datapool.data_keys[0]
        dataset = self.dataset
        self.celltype = celltype

        chr_name = variant_info['Chromosome'].values[0]
        variant_coord = variant_info['Start'].values[0]

        peak_info = self.dataset.datapool._query_peaks(
            celltype, chr_name, variant_coord - 4000000, variant_coord + 4000000).reset_index(drop=True).reset_index()
        self.data_key = data_key
        self.peak_info = peak_info
        self.chr_name = chr_name
        self.variant_coord = variant_coord

        track_start, track_end, variant_peak, peak_start, strand = self.get_peak_start_end_from_variant(
            variant_info, self.peak_info)
        self.peak_start = peak_start
        self.strand = strand
        self.track_start = track_start
        self.track_end = track_end
        self.variant_peak = np.unique(variant_peak)

    def get_peak_start_end_from_variant(self, variant_info, peak_info):
        """
        Determine the peak start and end positions for a specific variant.

        Parameters:
        - variant_info (DataFrame): Variant information DataFrame.
        - peak_info (DataFrame): Peak information DataFrame.

        Returns:
        - tuple: The start and end indices of the peak.
        """
        from pyranges import PyRanges as pr
        chr_name = variant_info['Chromosome'].values[0]
        variant_coord = variant_info['Start'].values[0]

        df = pr(peak_info.copy().reset_index()).join(pr(variant_info.copy()[['Chromosome', 'Start', 'End']]), how='left', suffix="_variant", apply_strand_suffix=False).df[[
            'index', 'Chromosome', 'Start', 'End', 'Expression_positive', 'Expression_negative', 'aTPM', 'TSS', 'Start_variant', 'End_variant']].set_index('index')
        variant_peak = df.query(
            'Start_variant>=Start & End_variant<=End').index.values

        if variant_peak.shape[0] == 0:
            raise ValueError(f"Variant not found in the peak information.")

        # Get the peak start and end positions based on n_peaks_upper_bound
        peak_start, peak_end = variant_peak.min() - self.dataset.n_peaks_upper_bound // 2, variant_peak.max() + \
            self.dataset.n_peaks_upper_bound // 2
        variant_peak = variant_peak - peak_start

        track_start = peak_info.iloc[peak_start].Start - self.dataset.padding
        track_end = peak_info.iloc[peak_end].End + self.dataset.padding

        # Assume strand is always positive for variants
        strand = 1

        return track_start, track_end, variant_peak, peak_start, strand

    def run_inference_for_variant(self, offset=0):
        """
        Run inference for a specific variant.
        """
        batch = self.dataset.datapool.generate_sample(
            self.chr_name, self.track_start, self.track_end, self.data_key, self.celltype, mut=self.mut)
        variant_peak = self.variant_peak - offset

        # Run the model inference
        inference_results, prepared_batch = self.run_inference_with_batch(
            batch)

        # Handle the inference results as needed
        return inference_results, prepared_batch, variant_peak
