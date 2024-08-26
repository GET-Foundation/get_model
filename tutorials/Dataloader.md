Here's a detailed documentation for the dataset classes in markdown format:

# Dataset Classes Documentation

## PretrainDataset

### 0. Purpose
The `PretrainDataset` class is designed for pretraining genomic models using cell type-specific ATAC-seq and gene expression data. It provides a flexible framework for loading and processing large-scale genomic data from Zarr files.

### 1. Input Data and Expected Format
- **Zarr directories**: Contains ATAC-seq peak data, gene expression data, and other genomic annotations.
- **Genome sequence data**: Zarr file containing reference genome sequence.
- **Insulation data**: Optional, for topologically associating domain (TAD) boundaries.
- **Hi-C data**: Optional, for chromatin interaction information.

Expected format for Zarr files:
- ATAC-seq peaks: Sparse matrix (nucleotides x features)
- Gene expression: Dense matrix (genes x cell types)
- Genome sequence: One-hot encoded (4 channels for A, C, G, T)

### 2. Subsetting Parameters
- `keep_celltypes`: List of cell types to include.
- `leave_out_celltypes`: List of cell types to exclude.
- `leave_out_chromosomes`: List of chromosomes to exclude.
- `peak_name`: Name of the peak track in Zarr files.
- `negative_peak_name`: Name of negative peak track (optional).
- `additional_peak_columns`: List of additional columns to include.

### 3. Augmentation Parameters
- `mask_ratio`: Proportion of peaks to mask during training.
- `random_shift_peak`: Random shift applied to peak positions.
- `peak_inactivation`: Method for simulating peak inactivation.
- `mutations`: Method for introducing mutations in sequences.

### 4. Other Parameters
- `n_peaks_lower_bound`: Minimum number of peaks per sample.
- `n_peaks_upper_bound`: Maximum number of peaks per sample.
- `max_peak_length`: Maximum allowed length for a peak.
- `center_expand_target`: Target length for centering and expanding peaks.
- `use_insulation`: Whether to use insulation data for sampling.
- `insulation_subsample_ratio`: Subsampling ratio for insulation data.
- `preload_count`: Number of windows to preload.
- `n_packs`: Number of data packs to preload for efficient data loading.

### 5. GetItem Logic
1. Select a window from preloaded data packs.
2. Extract ATAC-seq peaks, gene expression, and genomic sequence for the window.
3. Apply augmentations (masking, peak shifts, mutations) if specified.
4. Generate sample with:
   - ATAC-seq signal (sparse matrix)
   - Peak sequence (one-hot encoded)
   - Gene expression labels
   - Metadata (cell type, chromosome, coordinates)
5. Return the processed sample.

## InferenceDataset

### 0. Purpose
The `InferenceDataset` is designed for making predictions on specific genes or genomic regions using pretrained models. It allows for targeted analysis of gene regulation in different cell types.

### 1. Input Data and Expected Format
Same as `PretrainDataset`, with additional requirement:
- Gene annotation data (e.g., GENCODE): GTF file with gene coordinates and information.

### 2. Subsetting Parameters
- `gene_list`: List of genes to analyze.
- Other parameters same as `PretrainDataset`.

### 3. Augmentation Parameters
Same as `PretrainDataset`.

### 4. Other Parameters
- `assembly`: Genome assembly (e.g., "hg38", "mm10").
- `promoter_extend`: Base pairs to extend around gene promoters.

### 5. GetItem Logic
1. Select a gene and cell type from the gene-cell type pair list.
2. Identify the genomic window containing the gene.
3. Extract data (ATAC-seq, sequence, expression) for the window.
4. Center the data around the gene's transcription start site (TSS).
5. Apply any specified augmentations.
6. Return the processed sample with gene-specific metadata.

## PerturbationInferenceDataset

### 0. Purpose
This dataset is designed for analyzing the effects of genetic perturbations (e.g., mutations, peak inactivations) on gene regulation across different cell types.

### 1. Input Data and Expected Format
- Requires an `InferenceDataset` instance.
- Perturbation data: DataFrame with columns for chromosome, start, end, and perturbation type.

### 2. Subsetting Parameters
Inherited from `InferenceDataset`.

### 3. Augmentation Parameters
- `mode`: Type of perturbation ("mutation" or "peak_inactivation").

### 4. Other Parameters
Same as `InferenceDataset`.

### 5. GetItem Logic
1. Select a gene-cell type-perturbation combination.
2. Generate two samples using `InferenceDataset`:
   a. Wild-type sample
   b. Perturbed sample (applying the specified mutation or peak inactivation)
3. Return both samples as a pair for comparison.

## ReferenceRegionDataset

### 0. Purpose
The `ReferenceRegionDataset` is designed to incorporate motif information into the genomic data, allowing for analysis of transcription factor binding sites in relation to gene regulation.

### 1. Input Data and Expected Format
- Requires a `PretrainDataset` instance.
- Reference motif data: Zarr file containing motif scores for genomic regions.

### 2. Subsetting Parameters
Inherited from `PretrainDataset`.

### 3. Augmentation Parameters
- `leave_out_motifs`: List of motifs to exclude from analysis.

### 4. Other Parameters
- `motif_scaler`: Scaling factor for motif scores.
- `quantitative_atac`: Whether to use quantitative ATAC-seq signal.

### 5. GetItem Logic
1. Select a genomic window from the `PretrainDataset`.
2. Map the window to corresponding motif data.
3. Combine ATAC-seq, expression, and motif data.
4. Apply any masking or augmentations.
5. Return the integrated sample with motif information.

## InferenceReferenceRegionDataset

### 0. Purpose
This dataset combines the gene-specific focus of `InferenceDataset` with the motif integration of `ReferenceRegionDataset` for detailed analysis of gene regulation mechanisms.

### 1. Input Data and Expected Format
Same as `InferenceDataset` and `ReferenceRegionDataset`.

### 2. Subsetting Parameters
Inherited from both parent classes.

### 3. Augmentation Parameters
Inherited from both parent classes.

### 4. Other Parameters
Combination of parameters from `InferenceDataset` and `ReferenceRegionDataset`.

### 5. GetItem Logic
1. Select a gene and cell type using `InferenceDataset` logic.
2. Extract the genomic window containing the gene.
3. Integrate motif data for the window using `ReferenceRegionDataset` logic.
4. Center data around the gene's TSS.
5. Apply any specified augmentations.
6. Return the processed sample with gene-specific and motif-related information.

## PerturbationInferenceReferenceRegionDataset

### 0. Purpose
This dataset combines perturbation analysis with motif-aware gene regulation studies, allowing for in-depth investigation of how genetic alterations affect transcription factor binding and gene expression.

### 1. Input Data and Expected Format
- Requires an `InferenceReferenceRegionDataset` instance.
- Perturbation data: Same as `PerturbationInferenceDataset`.

### 2. Subsetting Parameters
Inherited from `InferenceReferenceRegionDataset`.

### 3. Augmentation Parameters
- `mode`: Type of perturbation ("mutation" or "peak_inactivation").

### 4. Other Parameters
Same as `InferenceReferenceRegionDataset`.

### 5. GetItem Logic
1. Select a gene-cell type-perturbation combination.
2. Generate two samples using `InferenceReferenceRegionDataset`:
   a. Wild-type sample with motif data
   b. Perturbed sample with motif data (applying the specified perturbation)
3. For peak inactivation, modify the motif scores in affected regions.
4. Return both samples as a pair, including motif information for comparison.

These dataset classes provide a comprehensive framework for analyzing gene regulation, integrating various genomic data types, and studying the effects of genetic perturbations across different cell types. They are designed to be flexible and can be adapted to different experimental designs and research questions in genomics and epigenomics.