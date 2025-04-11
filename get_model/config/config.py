import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from omegaconf import MISSING, OmegaConf

T = TypeVar("T")


def default_cfg():
    return field(default_factory=T)


def get_target_from_class_name(class_name: str) -> str:
    # TODO change this to the correct path
    return f"get_model.model.model.{class_name}"


def load_config(config_name, config_path="./"):
    # Initialize Hydra to load the configuration
    GlobalHydra.instance().clear()
    hydra.initialize(config_path=config_path, version_base="1.3")
    cfg = hydra.compose(config_name=config_name)
    return cfg


def export_config(cfg, yaml_file):
    OmegaConf.save(cfg, yaml_file)

def load_config_from_yaml(yaml_file):
    cfg = OmegaConf.load(yaml_file)
    return cfg

def pretty_print_config(cfg):
    logging.info(OmegaConf.to_yaml(cfg))


@dataclass
class ModelConfig(Generic[T]):
    """
    Configuration for the model.

    Attributes:
        _target_: The target path for the model class.
        cfg: The configuration for the model.
    """

    _target_: str = field(
        default_factory=lambda: get_target_from_class_name(T.__name__)
    )
    cfg: T = default_cfg()


@dataclass
class DatasetConfig:
    """
    Configuration for the dataset.

    Attributes:
        zarr_dirs: List of Zarr directory paths.
        n_peaks_lower_bound: Lower bound for number of peaks.
        n_peaks_upper_bound: Upper bound for number of peaks.
        max_peak_length: Maximum length of a peak.
        peak_count_filter: Filter for peak count.
        center_expand_target: Target for center expansion.
        padding: Padding value.
        peak_name: Name of the peak.
        negative_peak_name: Name of the negative peak.
        negative_peak_ratio: Ratio for negative peaks.
        additional_peak_columns: Additional columns for peaks.
        random_shift_peak: Random shift for peaks.
        use_insulation: Whether to use insulation.
        insulation_subsample_ratio: Subsample ratio for insulation.
        hic_path: Path to HiC data.
        preload_count: Number of items to preload.
        pin_mem: Whether to pin memory.
        n_packs: Number of packs.
        keep_celltypes: Celltypes to keep.
        leave_out_celltypes: Celltypes to leave out.
        leave_out_chromosomes: Chromosomes to leave out.
        non_redundant: Whether to use non-redundant data.
        filter_by_min_depth: Whether to filter by minimum depth.
        mask_ratio: Ratio for masking.
        peak_inactivation: Peak inactivation method.
        mutations: Mutation method.
        dataset_size: Size of the dataset.
        eval_dataset_size: Size of the evaluation dataset.
    """

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
    additional_peak_columns: list = field(
        default_factory=lambda: [
            "Expression_positive",
            "Expression_negative",
            "aTPM",
            "TSS",
        ]
    )
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
    keep_celltypes: Any | None = None
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
class RegionDatasetConfig:
    """
    Configuration for the region dataset.

    Attributes:
        root: Root directory for the dataset.
        metadata_path: Path to metadata.
        num_region_per_sample: Number of regions per sample.
        transform: Transformation to apply.
        data_type: Type of data.
        leave_out_celltypes: Celltypes to leave out.
        leave_out_chromosomes: Chromosomes to leave out.
        quantitative_atac: Whether to use quantitative ATAC.
        sampling_step: Step size for sampling.
        mask_ratio: Ratio for masking.
    """

    root: str = "/home/xf2217/Projects/new_finetune_data_all"
    metadata_path: str = "cell_type_align.txt"
    num_region_per_sample: int = 900
    transform: Optional[Any] = None
    data_type: str = "fetal"
    keep_celltypes: str | None = ""
    is_pretrain: bool = False
    leave_out_celltypes: str | None = "Astrocytes"
    leave_out_chromosomes: str | None = "chr4,chr14"
    quantitative_atac: bool = False
    sampling_step: int = 100
    mask_ratio: float = 0


@dataclass
class RegionMotifDatasetConfig:
    """
    Configuration for the region motif dataset.
    """

    zarr_path: str = MISSING
    celltypes: str = MISSING
    transform: Optional[Any] = None
    quantitative_atac: bool = False
    sampling_step: int = 50
    num_region_per_sample: int = 1000
    leave_out_chromosomes: str | None = None
    leave_out_celltypes: str | None = None
    mask_ratio: float = 0.0


@dataclass
class OptimizerConfig:
    """
    Configuration for the optimizer.

    Attributes:
        lr: Learning rate.
        min_lr: Minimum learning rate.
        weight_decay: Weight decay.
        opt: Optimizer type.
        opt_eps: Epsilon for optimizer.
        opt_betas: Beta values for optimizer.
    """

    lr: float = 0.001
    min_lr: float = 0.0001
    weight_decay: float = 0.05
    opt: str = "adamw"
    opt_eps: float = 1e-8
    opt_betas: list = field(default_factory=lambda: [0.9, 0.95])


@dataclass
class TrainingConfig:
    """
    Configuration for training.

    Attributes:
        save_ckpt_freq: Frequency of saving checkpoints.
        epochs: Number of epochs.
        warmup_epochs: Number of warmup epochs.
        accumulate_grad_batches: Number of batches to accumulate gradients.
        clip_grad: Gradient clipping value.
        use_fp16: Whether to use FP16.
        log_every_n_steps: Number of steps to log.
        val_check_interval: Validation check interval.
        save_every_n_epochs: Save checkpoint every n epochs, default is not saving.
    """

    save_ckpt_freq: int = 10
    epochs: int = 100
    warmup_epochs: int = 5
    accumulate_grad_batches: int = 1
    clip_grad: float | None = None
    use_fp16: bool = True
    log_every_n_steps: int = 25
    val_check_interval: float = 0.5
    add_lr_monitor: bool = False
    save_every_n_epochs: int | None = None

@dataclass
class RunConfig:
    """
    Configuration for the run.

    Attributes:
        project_name: Name of the project.
        run_name: Name of the run.
    """

    project_name: str = MISSING
    run_name: str = MISSING
    use_wandb: bool = True


@dataclass
class WandbConfig:
    """
    Obsolete, just for compatibility.
    Configuration for the run.

    Attributes:
        project_name: Name of the project.
        run_name: Name of the run.
    """

    project_name: str = MISSING
    run_name: str = MISSING
    use_wandb: bool = True


@dataclass
class FinetuneConfig:
    """
    Configuration for fine-tuning.

    Attributes:
        resume_ckpt: Checkpoint to resume from.
        pretrain_checkpoint: Whether to use pretrained checkpoint.
        checkpoint: Path to checkpoint.
        strict: Whether to use strict loading.
        model_key: Key for model in checkpoint.
        use_lora: Whether to use LoRA.
        lora_checkpoint: Path to LoRA checkpoint.
        rename_config: Configuration for renaming.
        layers_with_lora: Layers to apply LoRA.
        patterns_to_freeze: Patterns to freeze.
        patterns_to_drop: Patterns to drop.
        additional_checkpoints: Additional checkpoints.
    """

    resume_ckpt: str | None = None
    pretrain_checkpoint: bool = False
    checkpoint: str | None = None
    strict: bool = True
    model_key: str = "state_dict"
    use_lora: bool = False
    lora_checkpoint: str | None = None
    rename_config: dict | None = None
    layers_with_lora: list = field(
        default_factory=lambda: ["region_embed", "encoder", "head_exp"]
    )
    patterns_to_freeze: list = field(default_factory=lambda: ["motif_scanner"])
    patterns_to_drop: list = field(default_factory=lambda: [])
    additional_checkpoints: list = field(default_factory=lambda: [])


@dataclass
class MachineConfig:
    """
    Configuration for the machine.

    Attributes:
        codebase: Path to codebase.
        data_path: Path to data.
        output_dir: Output directory.
        num_devices: Number of devices.
        num_workers: Number of workers.
        batch_size: Batch size.
        fasta_path: Path to FASTA file.
    """

    codebase: str = MISSING
    data_path: str = MISSING
    output_dir: str = MISSING
    num_devices: int = 1
    num_workers: int = 32
    batch_size: int = 16
    fasta_path: str = MISSING


@dataclass
class TaskConfig:
    """
    Configuration for the task.

    Attributes:
        test_mode: Mode for testing.
        gene_list: List of genes.
        layer_names: Names of layers.
        mutations: Mutation method.
    """

    test_mode: str = "predict"
    gene_list: str | None = None
    layer_names: list = field(default_factory=lambda: ["atac_attention"])
    mutations: str | None = None


@dataclass
class Config:
    """
    Main configuration class.

    Attributes:
        run: Run configuration.
        type: Type of configuration.
        log_image: Whether to log images.
        stage: Stage of the process.
        assembly: Genome assembly.
        model: Model configuration.
        machine: Machine configuration.
        dataset: Dataset configuration.
        training: Training configuration.
        optimizer: Optimizer configuration.
        finetune: Fine-tuning configuration.
        task: Task configuration.
    """

    run: RunConfig = field(default_factory=RunConfig)
    type: str = "nucleotide"
    log_image: bool = False
    stage: str = "fit"
    assembly: str = "hg38"
    model: Any = MISSING
    machine: MachineConfig = field(default_factory=MachineConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    task: TaskConfig = field(default_factory=TaskConfig)


@dataclass
class RegionConfig:
    """
    Configuration for region.

    Attributes:
        run: Run configuration.
        type: Type of configuration.
        stage: Stage of the process.
        assembly: Genome assembly.
        eval_tss: Whether to evaluate TSS.
        log_image: Whether to log images.
        model: Model configuration.
        machine: Machine configuration.
        dataset: Dataset configuration for region.
        training: Training configuration.
        optimizer: Optimizer configuration.
        finetune: Fine-tuning configuration.
        task: Task configuration.
    """

    run: RunConfig = field(default_factory=RunConfig)
    type: str = "region"
    stage: str = "fit"
    assembly: str = "hg38"
    eval_tss: bool = False
    log_image: bool = False
    model: Any = MISSING
    machine: MachineConfig = field(default_factory=MachineConfig)
    dataset: RegionDatasetConfig = field(default_factory=RegionDatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    task: TaskConfig = field(default_factory=TaskConfig)


@dataclass
class RegionZarrConfig:
    """
    Configuration for region.

    Attributes:
        run: Run configuration.
        type: Type of configuration.
        stage: Stage of the process.
        assembly: Genome assembly.
        eval_tss: Whether to evaluate TSS.
        log_image: Whether to log images.
        model: Model configuration.
        machine: Machine configuration.
        dataset: Dataset configuration for region.
        training: Training configuration.
        optimizer: Optimizer configuration.
        finetune: Fine-tuning configuration.
        task: Task configuration.
    """

    run: RunConfig = field(default_factory=RunConfig)
    type: str = "region"
    stage: str = "fit"
    assembly: str = "hg38"
    eval_tss: bool = False
    log_image: bool = False
    model: Any = MISSING
    machine: MachineConfig = field(default_factory=MachineConfig)
    dataset: RegionMotifDatasetConfig = field(default_factory=RegionMotifDatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    task: TaskConfig = field(default_factory=TaskConfig)


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)

csr = ConfigStore.instance()
csr.store(name="base_region_config", node=RegionConfig)

csz = ConfigStore.instance()
csz.store(name="base_region_zarr_config", node=RegionZarrConfig)
