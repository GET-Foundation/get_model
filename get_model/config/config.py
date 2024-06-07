from ast import Dict
from dataclasses import dataclass, field
from email.policy import strict
from typing import Any, Generic, TypeVar

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from get_model.dataset.zarr_dataset import ReferenceRegionMotifConfig
from typing import Optional

T = TypeVar("T")


def default_cfg():
    return field(default_factory=T)


def get_target_from_class_name(class_name: str) -> str:
    # TODO change this to the correct path
    return f"get_model.model.model_refactored.{class_name}"


@dataclass
class ModelConfig(Generic[T]):
    _target_: str = field(
        default_factory=lambda: get_target_from_class_name(T.__name__))
    cfg: T = default_cfg()


@dataclass
class DatasetConfig:
    zarr_dirs: list = MISSING

    # peaks
    n_peaks_lower_bound: int = 5
    n_peaks_upper_bound: int = 10
    max_peak_length: int = 5000
    peak_count_filter: int = 0
    center_expand_target: int = 500
    padding: int = 0
    peak_name: str = "peaks_q0.01_tissue_open_exp"
    negative_peak_name: str | None = None
    negative_peak_ratio: float = 0
    additional_peak_columns: list = field(default_factory=lambda: [
        'Expression_positive', 'Expression_negative', 'aTPM', 'TSS'])
    random_shift_peak: int | None = None

    # insulation
    use_insulation: bool = False
    insulation_subsample_ratio: int = 1

    # hic
    hic_path: str | None = None

    # performance
    preload_count: int = 10
    pin_mem: bool = True
    n_packs: int = 1

    # leave-out & filtering
    keep_celltypes: str | None = None
    leave_out_celltypes: str | None = "Astrocyte"
    leave_out_chromosomes: str | None = "chr4,chr14"
    non_redundant: bool = False
    filter_by_min_depth: bool = False

    # Augmentation & perturbation
    mask_ratio: float = 0.5
    peak_inactivation: str | None = None
    mutations: str | None = None

    # Dataset size
    dataset_size: int = 40960
    eval_dataset_size: int = 4096


@dataclass
class ReferenceRegionDatasetConfig(DatasetConfig):
    reference_region_motif: ReferenceRegionMotifConfig = field(
        default_factory=ReferenceRegionMotifConfig)
    quantitative_atac: bool = False
    sampling_step: int = 100
    mask_ratio: float = 0


@dataclass
class RegionDatasetConfig:
    root: str = '/home/xf2217/Projects/new_finetune_data_all'
    metadata_path: str = 'cell_type_align.txt'
    num_region_per_sample: int = 900
    transform: Optional[Any] = None
    data_type: str = 'fetal'
    leave_out_celltypes: str | None = 'Astrocytes'
    leave_out_chromosomes: str | None = 'chr4,chr14'
    quantitative_atac: bool = False
    sampling_step: int = 100
    mask_ratio: float = 0


@dataclass
class OptimizerConfig:
    lr: float = 0.001
    min_lr: float = 0.0001
    weight_decay: float = 0.05
    opt: str = 'adamw'
    opt_eps: float = 1e-8
    opt_betas: list = field(default_factory=lambda: [0.9, 0.95])


@dataclass
class TrainingConfig:
    save_ckpt_freq: int = 10
    epochs: int = 100
    warmup_epochs: int = 5
    accumulate_grad_batches: int = 1
    clip_grad: float | None = None
    use_fp16: bool = True


@dataclass
class WandbConfig:
    project_name: str = "pretrain"
    run_name: str = "experiment_1"


@dataclass
class FinetuneConfig:
    pretrain_checkpoint: bool = False
    checkpoint: str | None = None
    strict: bool = True
    model_key: str = "state_dict"
    use_lora: bool = False
    lora_checkpoint: str | None = None
    rename_config: dict | None = None
    layers_with_lora: list = field(default_factory=lambda: ['region_embed', 'encoder', 'head_exp'])
    patterns_to_freeze: list = field(default_factory=lambda: [
        "motif_scanner"])
    patterns_to_drop: list = field(default_factory=lambda: [])
    additional_checkpoints: list = field(default_factory=lambda: [])

@dataclass
class MachineConfig:
    codebase: str = MISSING
    data_path: str = MISSING
    output_dir: str = MISSING
    num_devices: int = 1
    num_workers: int = 32
    batch_size: int = 16


@dataclass
class TaskConfig:
    test_mode: str = 'predict'
    gene_list: str | None = None
    layer_names: list = field(default_factory=lambda: ['atac_attention'])
    mutations: str | None = None


@dataclass
class Config:
    log_image: bool = False
    stage: str = 'fit'
    assembly: str = 'hg38'
    model: Any = MISSING
    machine: MachineConfig = field(default_factory=MachineConfig)
    dataset: DatasetConfig = field(
        default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    task: TaskConfig = field(
        default_factory=TaskConfig)


@dataclass
class ReferenceRegionConfig(Config):
    eval_tss: bool = False
    log_image: bool = False
    dataset: ReferenceRegionDatasetConfig = field(
        default_factory=ReferenceRegionDatasetConfig)


@dataclass
class RegionConfig:
    stage: str = 'fit'
    assembly: str = 'hg38'
    eval_tss: bool = False
    log_image: bool = False
    model: Any = MISSING
    machine: MachineConfig = field(default_factory=MachineConfig)
    dataset: RegionDatasetConfig = field(
        default_factory=RegionDatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    task: TaskConfig = field(
        default_factory=TaskConfig)


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)

csrr = ConfigStore.instance()
csrr.store(name="base_ref_region_config", node=ReferenceRegionConfig)

csr = ConfigStore.instance()
csr.store(name="base_region_config", node=RegionConfig)
