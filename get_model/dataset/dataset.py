import logging
import os
import os.path

from get_model.dataset.zarr_dataset import DenseZarrIO
from get_model.dataset.zarr_dataset import \
    PretrainDataset


def build_dataset_zarr_template(dataset_name, is_train, args, parameter_override=None, sequence_obj=None, root=None, codebase=None):
    """A template to build a dataset for training or evaluation."""
    logging.info(f'Using {dataset_name}')
    if root is None:
        root = args.data_path

    if codebase is None:
        codebase = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))

    if sequence_obj is None:
        sequence_obj = DenseZarrIO(f'{root}/hg38.zarr', dtype='int8')
        sequence_obj.load_to_memory_dense()
    else:
        logging.info('sequence_obj is provided')

    # Default parameter values
    parameters = {
        'zarr_dirs': [f'{root}/shendure_fetal_dense.zarr'],
        'genome_seq_zarr': f'{root}/hg38.zarr',
        'genome_motif_zarr': f'{root}/hg38_motif_result.zarr',
        'insulation_paths': [
            f'{codebase}/data/hg38_4DN_average_insulation.ctcf.adjecent.feather',
            f'{codebase}/data/hg38_4DN_average_insulation.ctcf.longrange.feather'],
        'peak_name': args.peak_name,
        'negative_peak_name': args.negative_peak_name,
        'additional_peak_columns': ['Expression_positive', 'Expression_negative', 'aTPM', 'TSS'],
        'preload_count': args.preload_count,
        'padding': args.padding,
        'mask_ratio': args.mask_ratio,
        'n_packs': args.n_packs,
        'max_peak_length': args.max_peak_length,
        'center_expand_target': args.center_expand_target,
        'insulation_subsample_ratio': args.insulation_subsample_ratio,
        'n_peaks_lower_bound': args.n_peaks_lower_bound,
        'n_peaks_upper_bound': args.n_peaks_upper_bound,
        'use_insulation': args.use_insulation,
        'random_shift_peak': args.random_shift_peak,
        'negative_peak_ratio': args.negative_peak_ratio,
        'peak_inactivation': args.peak_inactivation,
        'leave_out_celltypes': args.leave_out_celltypes,
        'leave_out_chromosomes': args.leave_out_chromosomes,
        'n_peaks_sample_gap': args.n_peaks_upper_bound,
        'non_redundant': args.non_redundant,
        'filter_by_min_depth': args.filter_by_min_depth,
        'dataset_size': args.dataset_size,
        'hic_path': args.hic_path
    }

    # Override parameters from parameter_override
    if parameter_override:
        for k, v in parameter_override.items():
            parameters[k] = v

    # Create dataset with updated parameters
    dataset = PretrainDataset(
        is_train=is_train,
        sequence_obj=sequence_obj,
        zarr_dirs=parameters['zarr_dirs'],
        genome_seq_zarr=parameters['genome_seq_zarr'],
        genome_motif_zarr=parameters['genome_motif_zarr'],
        use_insulation=parameters['use_insulation'],
        insulation_paths=parameters['insulation_paths'],
        insulation_subsample_ratio=parameters['insulation_subsample_ratio'],
        hic_path=parameters['hic_path'],

        peak_name=parameters['peak_name'],
        additional_peak_columns=parameters['additional_peak_columns'],
        max_peak_length=parameters['max_peak_length'],
        center_expand_target=parameters['center_expand_target'],
        n_peaks_lower_bound=parameters['n_peaks_lower_bound'],
        n_peaks_upper_bound=parameters['n_peaks_upper_bound'],
        n_peaks_sample_gap=parameters['n_peaks_sample_gap'],
        non_redundant=parameters['non_redundant'],
        filter_by_min_depth=parameters['filter_by_min_depth'],

        preload_count=parameters['preload_count'],
        n_packs=parameters['n_packs'],

        padding=parameters['padding'],
        mask_ratio=parameters['mask_ratio'],
        negative_peak_name=parameters['negative_peak_name'],
        negative_peak_ratio=parameters['negative_peak_ratio'],
        random_shift_peak=parameters['random_shift_peak'],
        peak_inactivation=parameters['peak_inactivation'],

        leave_out_celltypes=parameters['leave_out_celltypes'],
        leave_out_chromosomes=parameters['leave_out_chromosomes'],
        dataset_size=parameters['dataset_size'],
    )

    return dataset


def dataset_pretrain(is_train, args, sequence_obj=None):
    return build_dataset_zarr_template(
        "Pretrain", is_train, args, sequence_obj=sequence_obj, parameter_override={
            'zarr_dirs': [f'{args.data_path}/shendure_fetal_dense.zarr',
                          f'{args.data_path}/encode_hg38atac_dense.zarr',
                          f'{args.data_path}/vijay_hematopoiesis_dense.zarr',
                          f'{args.data_path}/htan_gbm_dense.zarr',
                          f'{args.data_path}/bingren_adult_dense.zarr',],
            'leave_out_celltypes': None,
            'additional_peak_columns': None,
        })


def dataset_pretrain_gbm_eval(is_train, args, sequence_obj=None):
    return build_dataset_zarr_template(
        "Pretrain.GBM_eval", is_train, args, sequence_obj=sequence_obj, parameter_override={
            'additional_peak_columns': None,
            'zarr_dirs': [
                f'{args.data_path}/htan_gbm_dense.zarr',
            ],
            'leave_out_celltypes': None,
            'dataset_size': 4096,
        })


def dataset_fintune_fetal(is_train, args, sequence_obj=None):
    return build_dataset_zarr_template(
        "Expression_Finetune_Fetal", False, args, sequence_obj=sequence_obj, parameter_override={
            'leave_out_celltypes': 'Astrocyte',
            'dataset_size': 4096,
        })


def dataset_fintune_fetal_eval(is_train, args, sequence_obj=None):
    return build_dataset_zarr_template(
        "Expression_Finetune_Fetal.fetal_eval", False, args, sequence_obj=sequence_obj, parameter_override={
            'leave_out_celltypes': 'Astrocyte',
            'dataset_size': 4096,
        })


def dataset_fintune_fetal_all_chr(is_train, args, sequence_obj=None):
    return build_dataset_zarr_template(
        "Expression_Finetune_Fetal.All_Chr", is_train, args, sequence_obj=sequence_obj, parameter_override={
            'leave_out_chromosomes': None
        })


def dataset_fintune_fetal_all_chr_eval(is_train, args, sequence_obj=None):
    return build_dataset_zarr_template(
        "Expression_Finetune_Fetal.All_Chr.eval", is_train, args, sequence_obj=sequence_obj, parameter_override={
            'dataset_size': 4096,
            'leave_out_chromosomes': None
        })


def dataset_fintune_fetal_k562_hsc(is_train, args, sequence_obj=None):
    return build_dataset_zarr_template(
        "Expression_Finetune_K562_HSC.Chr4&14", is_train, args, sequence_obj=sequence_obj, parameter_override={
            'zarr_dirs': [f'{args.data_path}/shendure_fetal_dense.zarr',
                          f'{args.data_path}/encode_hg38atac_dense.zarr',
                          f'{args.data_path}/vijay_hematopoiesis_dense.zarr'],
        })


def dataset_fintune_fetal_k562_hsc_eval(is_train, args, sequence_obj=None):
    return build_dataset_zarr_template(
        "Expression_Finetune_K562_HSC.Chr4&14.Eval", is_train, args, sequence_obj=sequence_obj, parameter_override={
            'zarr_dirs': [f'{args.data_path}/vijay_hematopoiesis_dense.zarr'],
            'dataset_size': 40_96,
        })


def dataset_fintune_monocyte(is_train, args, sequence_obj=None):
    return build_dataset_zarr_template(
        "Expression_Finetune_monocyte.Chr4&14", False, args, sequence_obj=sequence_obj, parameter_override={
            'zarr_dirs': [f'{args.data_path}/vijay_hematopoiesis_dense.zarr'],
            'leave_out_celltypes': "Mono.vijay_hematopoiesis.Young2_BMMC.1024",
            'leave_out_chromosomes': ['chr1', 'chr2', 'chr3', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX'],
        })


def dataset_fintune_monocyte_eval(is_train, args, sequence_obj=None):
    return build_dataset_zarr_template(
        "Expression_Finetune_monocyte.Chr4&14.Eval", False, args, sequence_obj=sequence_obj, parameter_override={
            'zarr_dirs': [f'{args.data_path}/vijay_hematopoiesis_dense.zarr'],
            'leave_out_celltypes': "Mono.vijay_hematopoiesis.Young2_BMMC.1024",
            'leave_out_chromosomes': ['chr4', 'chr14'],
            'dataset_size': 8192,
        })


def dataset_fintune_k562(is_train, args, sequence_obj=None):
    return build_dataset_zarr_template(
        "Expression_Finetune_k562.Chr4&14", False, args, sequence_obj=sequence_obj, parameter_override={
            'zarr_dirs': [f'{args.data_path}/encode_hg38atac_dense.zarr'],
            'leave_out_celltypes': "k562",
            'leave_out_chromosomes': ['chr1', 'chr2', 'chr3', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX'],
        })


def dataset_fintune_k562_eval(is_train, args, sequence_obj=None):
    return build_dataset_zarr_template(
        "Expression_Finetune_k562.Chr4&14.Eval", False, args, sequence_obj=sequence_obj, parameter_override={
            'zarr_dirs': [f'{args.data_path}/encode_hg38atac_dense.zarr'],
            'leave_out_celltypes': "k562",
            'leave_out_chromosomes': ['chr4', 'chr14'],
            'dataset_size': 8192,
        })


def dataset_fintune_thp1(is_train, args, sequence_obj=None):
    return build_dataset_zarr_template(
        "Expression_Finetune_THP1.Chr4&14", False, args, sequence_obj=sequence_obj, parameter_override={
            'zarr_dirs': [f'{args.data_path}/encode_hg38atac_dense.zarr'],
            'dataset_size': 81920,
            'additional_peak_columns': ['aTPM'],
            'leave_out_celltypes': "THP1",
            'leave_out_chromosomes': ['chr1', 'chr2', 'chr3', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX'],
        })


def dataset_fintune_thp1_eval(is_train, args, sequence_obj=None):
    return build_dataset_zarr_template(
        "Expression_Finetune_THP1.Chr4&14.Eval", False, args, sequence_obj=sequence_obj, parameter_override={
            'zarr_dirs': [f'{args.data_path}/encode_hg38atac_dense.zarr'],
            'additional_peak_columns': ['aTPM'],
            'leave_out_celltypes': "THP1",
            'leave_out_chromosomes': ['chr4', 'chr14'],
            'dataset_size': 8192,
            'random_shift_peak': None,
        })


def dataset_fintune_fetal_k562(is_train, args, sequence_obj=None):
    return build_dataset_zarr_template(
        "Expression_Finetune_K562.Chr4&14", is_train, args, sequence_obj=sequence_obj, parameter_override={
            'zarr_dirs': [
                f'{args.data_path}/encode_hg38atac_dense.zarr'],
            'dataset_size': 40_960,
        })


def dataset_fintune_fetal_k562_eval(is_train, args, sequence_obj=None):
    return build_dataset_zarr_template(
        "Expression_Finetune_K562.Chr4&14.Eval", is_train, args, sequence_obj=sequence_obj, parameter_override={
            'zarr_dirs': [f'{args.data_path}/encode_hg38atac_dense.zarr'],
            'dataset_size': 40_96,
        })


def dataset_htan_gbm(is_train, args, sequence_obj=None):
    return build_dataset_zarr_template(
        "HTAN_GBM", is_train, args, sequence_obj=sequence_obj, parameter_override={
            'dataset_size': 65_536,
        })


def dataset_htan_gbm_eval(is_train, args, sequence_obj=None):
    return build_dataset_zarr_template(
        "HTAN_GBM.eval", is_train, args, sequence_obj=sequence_obj, parameter_override={
            'dataset_size': 4096,
            'leave_out_chromosomes': None
        })


def dataset_htan_gbm_alb2281(is_train, args, sequence_obj=None):
    return build_dataset_zarr_template(
        "HTAN_GBM", is_train, args, sequence_obj=sequence_obj, parameter_override={
            'dataset_size': 65_536,
        }, root="/pmglocal/alb2281/get_resources", codebase="/pmglocal/alb2281/repos/get_model")


def dataset_htan_gbm_eval_alb2281(is_train, args, sequence_obj=None):
    return build_dataset_zarr_template(
        "HTAN_GBM.eval", is_train, args, sequence_obj=sequence_obj, parameter_override={
            'dataset_size': 4096,
            'leave_out_chromosomes': None
        }, root="/pmglocal/alb2281/get_resources", codebase="/pmglocal/alb2281/repos/get_model")


def build_dataset_zarr(is_train, args, sequence_obj=None):
    """
    Determines the appropriate dataset configuration based on training or evaluation mode and dataset type.
    """
    dataset_mapping = {
        ('Pretrain', True): dataset_pretrain,
        ('Pretrain.GBM_eval', False): dataset_pretrain_gbm_eval,
        ('Expression_Finetune_Fetal', True): dataset_fintune_fetal,
        ('Expression_Finetune_Fetal.fetal_eval', False): dataset_fintune_fetal_eval,
        ('HTAN_GBM', True): dataset_htan_gbm,
        ('HTAN_GBM.eval', False): dataset_htan_gbm_eval,
        ('HTAN_GBM.alb2281', True): dataset_htan_gbm_alb2281,
        ('HTAN_GBM.eval.alb2281', False): dataset_htan_gbm_eval_alb2281,
        ('Expression_Finetune_Fetal.All_Chr', True): dataset_fintune_fetal_all_chr,
        ('Expression_Finetune_Fetal.All_Chr.eval', False): dataset_fintune_fetal_all_chr_eval,
        ('Expression_Finetune_K562_HSC.Chr4&14', True): dataset_fintune_fetal_k562_hsc,
        ('Expression_Finetune_K562_HSC.Chr4&14.Eval', False): dataset_fintune_fetal_k562_hsc_eval,
        ('Expression_Finetune_K562.Chr4&14', True): dataset_fintune_fetal_k562,
        ('Expression_Finetune_K562.Chr4&14.Eval', False): dataset_fintune_fetal_k562_eval,
        ('Expression_Finetune_monocyte.Chr4&14', True): dataset_fintune_monocyte,
        ('Expression_Finetune_monocyte.Chr4&14.Eval', False): dataset_fintune_monocyte_eval,
        ('Expression_Finetune_k562.Chr4&14', True): dataset_fintune_k562,
        ('Expression_Finetune_k562.Chr4&14.Eval', False): dataset_fintune_k562_eval,
        ('Expression_Finetune_THP1.Chr4&14', True): dataset_fintune_thp1,
        ('Expression_Finetune_THP1.Chr4&14.Eval', False): dataset_fintune_thp1_eval,
    }

    dataset_key = (args.data_set if is_train else args.eval_data_set, is_train)
    dataset_function = dataset_mapping.get(dataset_key)

    if not dataset_function:
        raise NotImplementedError(
            "The specified dataset configuration is not implemented.")

    return dataset_function(is_train, args, sequence_obj)
