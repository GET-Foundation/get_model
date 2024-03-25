from ast import Dict
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from get_model.model.model_refactored import BaseGETModelConfig, GETChrombpNetBiasModelConfig

T = TypeVar("T")

def default_cfg():
    return field(default_factory=T)

def get_target_from_class_name(class_name: str) -> str:
    return f"get_model.model.model_refactored.{class_name}" #TODO change this to the correct path

@dataclass
class ModelConfig(Generic[T]):
    _target_: str = field(default_factory=lambda: get_target_from_class_name(T.__name__))
    cfg: T = default_cfg()

@dataclass
class DatasetConfig:
    data_set: str = "Expression_Finetune_Fetal"
    eval_data_set: str = "Expression_Finetune_Fetal.fetal_eval"
    batch_size: int = 16
    num_workers: int = 16
    n_peaks_lower_bound: int = 5
    n_peaks_upper_bound: int = 10
    max_peak_length: int = 5000
    center_expand_target: int = 500
    use_insulation: bool = False
    preload_count: int = 10
    random_shift_peak: int = 10
    pin_mem: bool = True
    peak_name: str = "peaks_q0.01_tissue_open_exp"
    negative_peak_name: str | None = None
    n_packs: int = 1
    leave_out_celltypes: str = "Astrocyte"
    leave_out_chromosomes: str = "chr4,chr14"
    additional_peak_columns: list = field(default_factory=lambda: [
                                          'Expression_positive', 'Expression_negative', 'aTPM', 'TSS'])
    padding: int = 0
    mask_ratio: float = 0.5
    insulation_subsample_ratio: int = 1
    negative_peak_ratio: int = 0
    peak_inactivation: str | None = None
    non_redundant: bool = False
    filter_by_min_depth: bool = False
    hic_path: str | None = None
    dataset_configs: dict = MISSING
    dataset_size: int = 40960
    eval_dataset_size: int = 4096



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
    checkpoint: str | None = None
    model_prefix: str = "model."
    patterns_to_freeze: list = field(default_factory=lambda: [
        "motif_scanner"])

@dataclass
class MachineConfig:
    codebase: str = MISSING
    data_path: str = MISSING
    output_dir: str = MISSING
    num_devices: int = 1

@dataclass
class Config:
    dataset_name: str = MISSING
    assembly: str = 'hg38'
    model: Any = MISSING
    machine: MachineConfig = field(default_factory=MachineConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    
cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)

