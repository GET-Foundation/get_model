import logging
import os
import os.path

from get_model.dataset.augmentation import (DataAugmentationForGETPeak,
                                            DataAugmentationForGETPeakFinetune)
from get_model.dataset.zarr_dataset import DenseZarrIO
from get_model.dataset.zarr_dataset import \
    PretrainDataset as ZarrPretrainDataset


def build_dataset_zarr(is_train, args, sequence_obj=None):
    if is_train and args.data_set == "Pretrain":
        transform = DataAugmentationForGETPeak(args)
        print("Data Aug = %s" % str(transform))
        root = args.data_path
        # get FILEPATH
        codebase = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        if sequence_obj is None:
            sequence_obj = DenseZarrIO(f'{root}/hg38.zarr', dtype='int8')
            sequence_obj.load_to_memory_dense()
        else:
            logging.info('sequence_obj is provided')
        dataset = ZarrPretrainDataset([f'{root}/shendure_fetal_dense.zarr'],
                                      f'{root}/hg38.zarr',
                                      f'{root}/hg38_motif_result.zarr', [
            f'{codebase}/data/hg38_4DN_average_insulation.ctcf.adjecent.feather',
            f'{codebase}/data/hg38_4DN_average_insulation.ctcf.longrange.feather'],
            peak_name=args.peak_name, preload_count=args.preload_count, insulation_subsample_ratio=0.8, n_packs=args.n_packs, max_peak_length=args.max_peak_length, center_expand_target=args.center_expand_target, n_peaks_lower_bound=args.n_peaks_lower_bound, n_peaks_upper_bound=args.n_peaks_upper_bound, use_insulation=args.use_insulation, sequence_obj=sequence_obj, leave_out_celltypes=args.leave_out_celltypes,
            leave_out_chromosomes=args.leave_out_chromosomes, is_train=is_train, dataset_size=40960, )
    elif not is_train and args.eval_data_set == "Pretrain.GBM_eval":
        transform = DataAugmentationForGETPeak(args)
        print("Data Aug = %s" % str(transform))
        root = args.data_path
        # get FILEPATH
        codebase = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        if sequence_obj is None:
            sequence_obj = DenseZarrIO(f'{root}/hg38.zarr', dtype='int8')
            sequence_obj.load_to_memory_dense()
        else:
            logging.info('sequence_obj is provided')
        dataset = ZarrPretrainDataset([
            f'{root}/htan_gbm_dense.zarr',
        ],
            f'{root}/hg38.zarr',
            f'{root}/hg38_motif_result.zarr', [
            f'{codebase}/data/hg38_4DN_average_insulation.ctcf.adjecent.feather',
            f'{codebase}/data/hg38_4DN_average_insulation.ctcf.longrange.feather'],
            peak_name=args.peak_name, preload_count=args.preload_count, insulation_subsample_ratio=0.8, n_packs=args.n_packs, max_peak_length=args.max_peak_length, center_expand_target=args.center_expand_target, n_peaks_lower_bound=args.n_peaks_lower_bound, n_peaks_upper_bound=args.n_peaks_upper_bound, use_insulation=args.use_insulation, sequence_obj=sequence_obj, leave_out_celltypes=args.leave_out_celltypes,
            leave_out_chromosomes=args.leave_out_chromosomes,  is_train=is_train, dataset_size=4096)
    elif is_train and args.data_set == "Expression_Finetune_Fetal":
        transform = DataAugmentationForGETPeak(args)
        print("Data Aug = %s" % str(transform))
        root = args.data_path
        # get FILEPATH
        codebase = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        if sequence_obj is None:
            sequence_obj = DenseZarrIO(f'{root}/hg38.zarr', dtype='int8')
            sequence_obj.load_to_memory_dense()
        else:
            logging.info('sequence_obj is provided')
        logging.info('Using Expression_Finetune_Fetal')
        dataset = ZarrPretrainDataset([
            f'{root}/shendure_fetal_dense.zarr',
            f'{root}/htan_gbm_dense.zarr',
            f'{root}/vijay_hematopoiesis_dense.zarr',
        ],
            f'{root}/hg38.zarr',
            f'{root}/hg38_motif_result.zarr', [
            f'{codebase}/data/hg38_4DN_average_insulation.ctcf.adjecent.feather',
            f'{codebase}/data/hg38_4DN_average_insulation.ctcf.longrange.feather'],
            peak_name=args.peak_name, insulation_subsample_ratio=0.8,
            additional_peak_columns=['Expression_positive', 'Expression_negative', 'aTPM', 'TSS'], preload_count=args.preload_count,
            n_packs=args.n_packs, max_peak_length=args.max_peak_length, center_expand_target=args.center_expand_target,
            n_peaks_lower_bound=args.n_peaks_lower_bound, n_peaks_upper_bound=args.n_peaks_upper_bound, use_insulation=args.use_insulation,
            sequence_obj=sequence_obj, leave_out_celltypes=args.leave_out_celltypes, leave_out_chromosomes=None,
            is_train=is_train, non_redundant=args.non_redundant, filter_by_min_depth=args.filter_by_min_depth, dataset_size=40960)
    elif not is_train and args.eval_data_set == "Expression_Finetune_Fetal.fetal_eval":
        transform = DataAugmentationForGETPeak(args)
        print("Data Aug = %s" % str(transform))
        root = args.data_path
        # get FILEPATH
        codebase = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        if sequence_obj is None:
            sequence_obj = DenseZarrIO(f'{root}/hg38.zarr', dtype='int8')
            sequence_obj.load_to_memory_dense()
        else:
            logging.info('sequence_obj is provided')

        logging.info('Using Expression_Finetune_Fetal.fetal_eval')
        dataset = ZarrPretrainDataset([
            f'{root}/shendure_fetal_dense.zarr',
        ],
            f'{root}/hg38.zarr',
            f'{root}/hg38_motif_result.zarr', [
            f'{codebase}/data/hg38_4DN_average_insulation.ctcf.adjecent.feather',
            f'{codebase}/data/hg38_4DN_average_insulation.ctcf.longrange.feather'],
            peak_name=args.peak_name, insulation_subsample_ratio=0.8, additional_peak_columns=['Expression_positive', 'Expression_negative', 'aTPM', 'TSS'], preload_count=args.preload_count,
            n_packs=args.n_packs, max_peak_length=args.max_peak_length, center_expand_target=args.center_expand_target, n_peaks_lower_bound=args.n_peaks_lower_bound, n_peaks_upper_bound=args.n_peaks_upper_bound, use_insulation=args.use_insulation, sequence_obj=sequence_obj, leave_out_celltypes=args.leave_out_celltypes, leave_out_chromosomes=None, is_train=is_train, non_redundant=None, filter_by_min_depth=None, dataset_size=4096)
    elif is_train and args.data_set == "HTAN_GBM":
        transform = DataAugmentationForGETPeak(args)
        print("Data Aug = %s" % str(transform))
        root = args.data_path
        # get FILEPATH
        codebase = "/pmglocal/alb2281/repos/get_model"
        resource_dir = "/pmglocal/alb2281/get_resources"
        if sequence_obj is None:
            sequence_obj = DenseZarrIO(f'{root}/hg38.zarr', dtype='int8')
            sequence_obj.load_to_memory_dense()
        else:
            logging.info('sequence_obj is provided')
        logging.info('Using HTAN_GBM')
        dataset = ZarrPretrainDataset([
            f'{root}/htan_gbm_dense.zarr',
        ],
            f'{resource_dir}/hg38.zarr',
            f'{resource_dir}/hg38_motif_result.zarr', [
            f'{codebase}/data/hg38_4DN_average_insulation.ctcf.adjecent.feather',
            f'{codebase}/data/hg38_4DN_average_insulation.ctcf.longrange.feather'],
            peak_name=args.peak_name, insulation_subsample_ratio=0.8,
            additional_peak_columns=['Expression_positive', 'Expression_negative', 'aTPM', 'TSS'], preload_count=args.preload_count,
            n_packs=args.n_packs, max_peak_length=args.max_peak_length, center_expand_target=args.center_expand_target,
            n_peaks_lower_bound=args.n_peaks_lower_bound, n_peaks_upper_bound=args.n_peaks_upper_bound, use_insulation=args.use_insulation,
            sequence_obj=sequence_obj, leave_out_celltypes=args.leave_out_celltypes, leave_out_chromosomes=args.leave_out_chromosomes,
            is_train=is_train, non_redundant=args.non_redundant, filter_by_min_depth=args.filter_by_min_depth, dataset_size=65536)
    elif not is_train and args.eval_data_set == "HTAN_GBM.eval":
        transform = DataAugmentationForGETPeak(args)
        print("Data Aug = %s" % str(transform))
        root = args.data_path
        # get FILEPATH
        codebase = "/pmglocal/alb2281/repos/get_model"
        resource_dir = "/pmglocal/alb2281/get_resources"
        if sequence_obj is None:
            sequence_obj = DenseZarrIO(f'{root}/hg38.zarr', dtype='int8')
            sequence_obj.load_to_memory_dense()
        else:
            logging.info('sequence_obj is provided')

        logging.info('Using HTAN_GBM.eval')
        dataset = ZarrPretrainDataset([
            f"{root}/htan_gbm_dense.zarr",
        ],
            f'{resource_dir}/hg38.zarr',
            f'{resource_dir}/hg38_motif_result.zarr', [
            f'{codebase}/data/hg38_4DN_average_insulation.ctcf.adjecent.feather',
            f'{codebase}/data/hg38_4DN_average_insulation.ctcf.longrange.feather'],
            peak_name=args.peak_name, insulation_subsample_ratio=0.8, additional_peak_columns=['Expression_positive', 'Expression_negative', 'aTPM', 'TSS'],
            preload_count=args.preload_count,
            n_packs=args.n_packs, max_peak_length=args.max_peak_length, center_expand_target=args.center_expand_target, n_peaks_lower_bound=args.n_peaks_lower_bound,
            n_peaks_upper_bound=args.n_peaks_upper_bound, use_insulation=args.use_insulation, sequence_obj=sequence_obj, leave_out_celltypes=args.leave_out_celltypes,
            leave_out_chromosomes=args.leave_out_chromosomes, is_train=False, non_redundant=args.non_redundant, filter_by_min_depth=args.filter_by_min_depth, dataset_size=4096)
    else:
        raise NotImplementedError()

    return dataset
