import logging
import os.path

import hydra
import lightning as L
import torch
import torch.utils.data
from caesar.io.zarr_io import DenseZarrIO
from hydra.utils import instantiate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins import MixedPrecision
from lightning.pytorch.utilities import grad_norm
from omegaconf import MISSING, DictConfig

from get_model.config.config import *
from get_model.dataset.collate import get_rev_collate_fn
from get_model.dataset.zarr_dataset import DenseZarrIO, PretrainDataset
from get_model.model.model_refactored import *
from get_model.model.modules import *
from get_model.optim import create_optimizer
from get_model.utils import cosine_scheduler, load_checkpoint, remove_keys



class LitModel(L.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model = self.get_model()
        self.loss = self.model.loss
        self.metrics = self.model.metrics
        self.save_hyperparameters()

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

    def get_model(self):
        model = instantiate(self.cfg.model)
        if self.cfg.finetune.checkpoint is not None:
            checkpoint_model = load_checkpoint(
                self.cfg.finetune.checkpoint)
            state_dict = model.state_dict()
            remove_keys(checkpoint_model, state_dict)
            # checkpoint_model = rename_keys(checkpoint_model)
            model.load_state_dict(checkpoint_model, strict=False)
        model.freeze_layers(
            patterns_to_freeze=self.cfg.finetune.patterns_to_freeze, invert_match=False)
        return model

    def forward(self, batch):
        return self.model(**batch)

    def _shared_step(self, batch, batch_idx, stage='train'):
        batch = self.model.get_input(batch)
        output = self(batch)
        pred, obs = self.model.before_loss(output, batch)
        loss = self.loss(pred, obs)
        # if loss is a dict, rename the keys with the stage prefix
        if isinstance(loss, dict):
            loss = {f"{stage}_{key}": value for key,
                    value in loss.items()}
            self.log_dict(loss, batch_size=self.cfg.dataset.batch_size)
        loss = self.model.after_loss(loss)
        return loss, pred, obs

    def training_step(self, batch, batch_idx):
        loss, pred, obs = self._shared_step(batch, batch_idx, stage='train')
        self.log("train_loss", loss, batch_size=self.cfg.dataset.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred, obs = self._shared_step(batch, batch_idx, stage='val')
        metrics = self.metrics(pred, obs)
        self.log_dict(metrics, batch_size=self.cfg.dataset.batch_size)
        self.log("val_loss", loss, batch_size=self.cfg.dataset.batch_size)

    def configure_optimizers(self):
        optimizer = create_optimizer(self.cfg.optimizer, self.model)
        num_training_steps_per_epoch = (
            self.cfg.dataset.dataset_size // self.cfg.dataset.batch_size // self.cfg.training.num_devices
        )
        schedule = cosine_scheduler(
            base_value=self.cfg.optimizer.lr,
            final_value=self.cfg.optimizer.min_lr,
            epochs=self.cfg.training.epochs,
            niter_per_ep=num_training_steps_per_epoch,
            warmup_epochs=self.cfg.training.warmup_epochs,
            start_warmup_value=0,
            warmup_steps=-1
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: schedule[step],
        )
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        sequence_obj = DenseZarrIO(
            f'{self.cfg.dataset.data_path}/hg38.zarr', dtype='int8')
        sequence_obj.load_to_memory_dense()
        dataset_train = build_dataset_zarr(
            self.cfg.dataset, sequence_obj=sequence_obj)
        return torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self.cfg.dataset.batch_size,
            num_workers=self.cfg.dataset.num_workers,
            drop_last=True,
            collate_fn=get_rev_collate_fn,
        )

    def val_dataloader(self):
        sequence_obj = DenseZarrIO(
            f'{self.cfg.dataset.data_path}/hg38.zarr', dtype='int8')
        sequence_obj.load_to_memory_dense()
        dataset_eval = build_dataset_zarr(
            args=self.cfg.dataset, sequence_obj=sequence_obj)
        return torch.utils.data.DataLoader(
            dataset_eval,
            batch_size=self.cfg.dataset.batch_size,
            num_workers=self.cfg.dataset.num_workers,
            drop_last=True,
            collate_fn=get_rev_collate_fn
        )


class GETDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def build_dataset_zarr(self, sequence_obj) -> None:
        config = self.cfg.dataset.dataset_configs[self.cfg.dataset_name]
        dataset_name = config.data_set if config.is_train else config.eval_data_set
        logging.info(f'Using {dataset_name}')

        root = config.data_path
        codebase = config.codebase
        assembly = config.assembly

        if sequence_obj is None:
            sequence_obj = DenseZarrIO(f'{root}/{assembly}.zarr', dtype='int8')
            sequence_obj.load_to_memory_dense()
        else:
            logging.info('sequence_obj is provided')
            sequence_obj = sequence_obj

        # Create dataset with configuration parameters
        dataset = PretrainDataset(
            is_train=config.is_train,
            sequence_obj=sequence_obj,
            zarr_dirs=config.zarr_dirs,
            genome_seq_zarr=f'{root}/{assembly}.zarr',
            genome_motif_zarr=f'{root}/{assembly}_motif_result.zarr',
            use_insulation=config.use_insulation,
            insulation_paths=[
                f'{codebase}/data/{assembly}_4DN_average_insulation.ctcf.adjecent.feather',
                f'{codebase}/data/{assembly}_4DN_average_insulation.ctcf.longrange.feather'],
            hic_path=config.hic_path,

            peak_name=config.peak_name,
            additional_peak_columns=config.additional_peak_columns,
            max_peak_length=config.max_peak_length,
            center_expand_target=config.center_expand_target,
            n_peaks_lower_bound=config.n_peaks_lower_bound,
            n_peaks_upper_bound=config.n_peaks_upper_bound,
            n_peaks_sample_gap=config.n_peaks_upper_bound,
            non_redundant=config.non_redundant,
            filter_by_min_depth=config.filter_by_min_depth,

            preload_count=config.preload_count,
            n_packs=config.n_packs,

            padding=config.padding,
            mask_ratio=config.mask_ratio,
            negative_peak_name=config.negative_peak_name,
            negative_peak_ratio=config.negative_peak_ratio,
            random_shift_peak=config.random_shift_peak,
            peak_inactivation=config.peak_inactivation,

            leave_out_celltypes=config.leave_out_celltypes,
            leave_out_chromosomes=config.leave_out_chromosomes,
            dataset_size=config.dataset_size,
        )

        return dataset


    def prepare_data(self):
        pass

    def setup(self, stage=None):
        sequence_obj = DenseZarrIO(
            f'{self.cfg.dataset.data_path}/{self.cfg.assembly}.zarr', dtype='int8')
        sequence_obj.load_to_memory_dense()
        if stage == 'fit' or stage is None:
            self.dataset_train = self.build_dataset_zarr(sequence_obj=sequence_obj)
            self.dataset_val = self.build_dataset_zarr(sequence_obj=sequence_obj)
        if stage == 'test' or stage is None:
            self.dataset_test = self.build_dataset_zarr(sequence_obj=sequence_obj)
        if stage == 'predict' or stage is None:
            self.dataset_predict = self.build_dataset_zarr(sequence_obj=sequence_obj)
        if stage == 'validate' or stage is None:
            self.dataset_val = self.build_dataset_zarr(sequence_obj=sequence_obj)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.cfg.dataset.batch_size,
            num_workers=self.cfg.dataset.num_workers,
            drop_last=True,
            collate_fn=get_rev_collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_val,
            batch_size=self.cfg.dataset.batch_size,
            num_workers=self.cfg.dataset.num_workers,
            drop_last=True,
            collate_fn=get_rev_collate_fn,
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_test,
            batch_size=self.cfg.dataset.batch_size,
            num_workers=self.cfg.dataset.num_workers,
            drop_last=True,
            collate_fn=get_rev_collate_fn,
        )
    
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_predict,
            batch_size=self.cfg.dataset.batch_size,
            num_workers=self.cfg.dataset.num_workers,
            drop_last=True,
            collate_fn=get_rev_collate_fn,
        )
    

class GETVariantInferenceDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.dataset = None
        self.model = None
        self.variant_info = None
        self.chr_name = None
        self.variant_coord = None
        self.peak_info = None
        self.track_start = None
        self.track_end = None
        self.variant_peak = None

    def setup(self, stage=None):
        if stage == 'predict' or stage is None:
            sequence_obj = DenseZarrIO(f'{self.cfg.dataset.data_path}/{self.cfg.assembly}.zarr', dtype='int8')
            sequence_obj.load_to_memory_dense()
            self.dataset = instantiate(self.cfg.dataset, sequence_obj=sequence_obj)
            self.model = instantiate(self.cfg.model)

            self.variant_info = pd.DataFrame(self.cfg.variant_info)
            self.chr_name = self.variant_info['Chromosome'].values[0]
            self.variant_coord = self.variant_info['Start'].values[0]

            peak_info = self.dataset.datapool._query_peaks(
                self.cfg.cell_type, self.chr_name, self.variant_coord - 4000000, self.variant_coord + 4000000
            ).reset_index(drop=True).reset_index()
            self.peak_info = peak_info

            self.track_start, self.track_end, self.variant_peak = self.get_peak_start_end_from_variant(
                self.variant_info, self.peak_info
            )

    def get_peak_start_end_from_variant(self, variant_info, peak_info):
        from pyranges import PyRanges as pr
        chr_name = variant_info['Chromosome'].values[0]
        variant_coord = variant_info['Start'].values[0]

        df = pr(peak_info.copy().reset_index()).join(
            pr(variant_info.copy()[['Chromosome', 'Start', 'End']]),
            how='left', suffix="_variant", apply_strand_suffix=False
        ).df[['index', 'Chromosome', 'Start', 'End', 'Start_variant', 'End_variant']].set_index('index')
        variant_peak = df.query('Start_variant>=Start & End_variant<=End').index.values

        if variant_peak.shape[0] == 0:
            raise ValueError(f"Variant not found in the peak information.")

        peak_start = max(0, variant_peak.min() - self.dataset.n_peaks_upper_bound // 2)
        peak_end = min(peak_info.shape[0] - 1, variant_peak.max() + self.dataset.n_peaks_upper_bound // 2)
        variant_peak = variant_peak - peak_start

        track_start = peak_info.iloc[peak_start].Start - self.dataset.padding
        track_end = peak_info.iloc[peak_end].End + self.dataset.padding

        return track_start, track_end, variant_peak

    def predict_dataloader(self):
        batch = self.dataset.datapool.generate_sample(
            self.chr_name, self.track_start, self.track_end, self.dataset.datapool.data_keys[0],
            self.cfg.cell_type, mut=self.cfg.mut
        )
        prepared_batch = get_rev_collate_fn([batch])
        return [prepared_batch]


def run(cfg: DictConfig):
    torch.set_float32_matmul_precision('medium')
    model = LitModel(cfg)
    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="gpu",
        num_sanity_val_steps=10,
        strategy="auto",
        devices=cfg.training.num_devices,
        logger=[WandbLogger(project=cfg.wandb.project_name,
                            name=cfg.wandb.run_name)],
        callbacks=[ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, save_last=True, filename="best"),
                   LearningRateMonitor(logging_interval='step')],
        plugins=[MixedPrecision(precision='16-mixed', device="cuda")],
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        gradient_clip_val=cfg.training.clip_grad,
        log_every_n_steps=100,
        deterministic=True,
        default_root_dir=cfg.training.output_dir,
    )

    trainer.fit(model)
