import logging
import os
import os.path

from get_model.dataset.augmentation import (DataAugmentationForGETPeak,
                                            DataAugmentationForGETPeakFinetune)
from get_model.dataset.zarr_dataset import DenseZarrIO
from get_model.dataset.zarr_dataset import \
    PretrainDataset as ZarrPretrainDataset

def build_dataset_zarr_template(dataset_name, is_train, args, parameter_override=None, sequence_obj=None, root=None, codebase=None):
    """A template to build a dataset for training or evaluation."""
    logging.info(f'Using {dataset_name}')
    transform = DataAugmentationForGETPeak(args)
    print("Data Aug = %s" % str(transform))
    if root==None:
        root = args.data_path
    # get FILEPATH
    if codebase==None:
        codebase = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    if sequence_obj is None:
        sequence_obj = DenseZarrIO(f'{root}/hg38.zarr', dtype='int8')
        sequence_obj.load_to_memory_dense()
    else:
        logging.info('sequence_obj is provided')
    zarr_dirs = [f'{root}/shendure_fetal_dense.zarr']
    genome_seq_zarr = f'{root}/hg38.zarr'
    genome_motif_zarr = f'{root}/hg38_motif_result.zarr'
    insulation_paths = [
        f'{codebase}/data/hg38_4DN_average_insulation.ctcf.adjecent.feather',
        f'{codebase}/data/hg38_4DN_average_insulation.ctcf.longrange.feather']
    peak_name=args.peak_name
    additional_peak_columns=['Expression_positive', 'Expression_negative', 'aTPM', 'TSS']
    preload_count=args.preload_count
    padding=50
    mask_ratio=0.5
    n_packs=args.n_packs
    max_peak_length=args.max_peak_length
    center_expand_target=args.center_expand_target
    insulation_subsample_ratio=0.8
    n_peaks_lower_bound=args.n_peaks_lower_bound
    n_peaks_upper_bound=args.n_peaks_upper_bound
    use_insulation=args.use_insulation
    leave_out_celltypes=args.leave_out_celltypes
    leave_out_chromosomes=args.leave_out_chromosomes
    n_peaks_sample_gap=50
    non_redundant=args.non_redundant
    filter_by_min_depth=args.filter_by_min_depth
    dataset_size=40_960
    # override parameters above from parameter_override
    if parameter_override is not None:
        for k, v in parameter_override.items():
            exec(f'{k} = {v}')
    dataset = ZarrPretrainDataset(zarr_dirs, genome_seq_zarr, genome_motif_zarr, insulation_paths,
        peak_name=peak_name, preload_count=preload_count, insulation_subsample_ratio=insulation_subsample_ratio, n_packs=n_packs, max_peak_length=max_peak_length, center_expand_target=center_expand_target, 
        padding=padding, mask_ratio=mask_ratio, 
        n_peaks_lower_bound=n_peaks_lower_bound, n_peaks_upper_bound=n_peaks_upper_bound, additional_peak_columns=additional_peak_columns, 
        n_peaks_sample_gap=n_peaks_sample_gap, non_redundant=non_redundant, filter_by_min_depth=filter_by_min_depth,
        use_insulation=use_insulation, sequence_obj=sequence_obj, leave_out_celltypes=leave_out_celltypes,
        leave_out_chromosomes=leave_out_chromosomes, is_train=is_train, dataset_size=dataset_size)
    return dataset

def dataset_pretrain(is_train, args, sequence_obj=None):
    return build_dataset_zarr_template(
        "Pretrain", is_train, args, sequence_obj=sequence_obj, parameter_override={
        'dataset_size': 40_960,
        })

def dataset_pretrain_gbm_eval(is_train, args, sequence_obj=None):
    return build_dataset_zarr_template(
        "Pretrain.GBM_eval", is_train, args, sequence_obj=sequence_obj, parameter_override={
        'dataset_size': 4096,
        'leave_out_celltypes': 'Astrocytes',
        'leave_out_chromosomes': None
        })


def dataset_fintune_fetal(is_train, args, sequence_obj=None):
    return build_dataset_zarr_template(
        "Expression_Finetune_Fetal", is_train, args, sequence_obj=sequence_obj, parameter_override={
        'dataset_size': 40_960,
        })

def dataset_fintune_fetal_eval(is_train, args, sequence_obj=None):
    return build_dataset_zarr_template(
        "Expression_Finetune_Fetal.fetal_eval", is_train, args, sequence_obj=sequence_obj, parameter_override={
        'dataset_size': 4096,
        })

def dataset_fintune_fetal_all_chr(is_train, args, sequence_obj=None):
    return build_dataset_zarr_template(
        "Expression_Finetune_Fetal.All_Chr", is_train, args, sequence_obj=sequence_obj, parameter_override={
        'dataset_size': 40_960,
        'leave_out_chromosomes': None
        })

def dataset_fintune_fetal_all_chr_eval(is_train, args, sequence_obj=None):
    return build_dataset_zarr_template(
        "Expression_Finetune_Fetal.All_Chr.eval", is_train, args, sequence_obj=sequence_obj, parameter_override={
        'dataset_size': 4096,
        'leave_out_chromosomes': None
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
    if is_train and args.data_set == "Pretrain":
        dataset = dataset_pretrain(is_train, args, sequence_obj)
    elif not is_train and args.eval_data_set == "Pretrain.GBM_eval":
        dataset = dataset_pretrain_gbm_eval(is_train, args, sequence_obj)
    elif is_train and args.data_set == "Expression_Finetune_Fetal":
        dataset = dataset_fintune_fetal(is_train, args, sequence_obj)
    elif not is_train and args.eval_data_set == "Expression_Finetune_Fetal.fetal_eval":
        dataset = dataset_fintune_fetal_eval(is_train, args, sequence_obj)
    elif is_train and args.data_set == "HTAN_GBM":
        dataset = dataset_htan_gbm(is_train, args, sequence_obj)
    elif not is_train and args.eval_data_set == "HTAN_GBM.eval":
        dataset = dataset_htan_gbm_eval(is_train, args, sequence_obj)
    elif is_train and args.data_set == "HTAN_GBM.alb2281":
        dataset = dataset_htan_gbm_alb2281(is_train, args, sequence_obj)
    elif not is_train and args.eval_data_set == "HTAN_GBM.eval.alb2281":
        dataset = dataset_htan_gbm_eval_alb2281(is_train, args, sequence_obj)
    elif is_train and args.data_set == "Expression_Finetune_Fetal.All_Chr":
        dataset = dataset_fintune_fetal_all_chr(is_train, args, sequence_obj)
    elif not is_train and args.eval_data_set == "Expression_Finetune_Fetal.All_Chr.eval":
        dataset = dataset_fintune_fetal_all_chr_eval(is_train, args, sequence_obj)
    else:
        raise NotImplementedError()

    return dataset
