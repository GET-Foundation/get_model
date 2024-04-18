import logging

import lightning as L
import pandas as pd
import torch
import torch.utils.data
from hydra.utils import instantiate
from lightning.pytorch.callbacks import (Callback, LearningRateMonitor,
                                         ModelCheckpoint)
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.plugins import MixedPrecision
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.utilities import grad_norm
from omegaconf import MISSING, DictConfig, OmegaConf
from tqdm import tqdm

from get_model.config.config import *
from get_model.dataset.collate import (get_perturb_collate_fn,
                                       get_rev_collate_fn)
from get_model.dataset.zarr_dataset import (InferenceDataset,
                                            PerturbationInferenceDataset,
                                            PretrainDataset, get_gencode_obj,
                                            get_sequence_obj)
from get_model.model.model_refactored import *
from get_model.model.modules import *
from get_model.optim import create_optimizer
from get_model.utils import cosine_scheduler, load_checkpoint, remove_keys, rename_lit_state_dict

logging.disable(logging.WARN)


class LitModel(L.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model = self.get_model()
        self.loss = self.model.loss
        self.metrics = self.model.metrics
        self.lr = cfg.optimizer.lr
        self.save_hyperparameters()
        self.dm = None

    # def on_before_optimizer_step(self, optimizer):
    #     # Compute the 2-norm for each layer
    #     # If using mixed precision, the gradients are already unscaled here
    #     # norms = grad_norm(self.model, norm_type=2)
    #     # self.log_dict(norms)

    def get_model(self):
        model = instantiate(self.cfg.model)
        if self.cfg.finetune.checkpoint is not None:
            checkpoint_model = load_checkpoint(
                self.cfg.finetune.checkpoint)
        checkpoint_model = rename_lit_state_dict(checkpoint_model)
        model.load_state_dict(checkpoint_model, strict=True)
        model.freeze_layers(
            patterns_to_freeze=self.cfg.finetune.patterns_to_freeze, invert_match=False)
        return model

    def forward(self, batch):
        return self.model(**batch)

    def _shared_step(self, batch, batch_idx, stage='train'):
        input = self.model.get_input(batch)
        output = self(input)
        pred, obs = self.model.before_loss(output, batch)
        loss = self.loss(pred, obs)
        # if loss is a dict, rename the keys with the stage prefix
        if isinstance(loss, dict):
            loss = {f"{stage}_{key}": value for key,
                    value in loss.items()}
            self.log_dict(loss, batch_size=self.cfg.machine.batch_size)
        loss = self.model.after_loss(loss)
        return loss, pred, obs

    def training_step(self, batch, batch_idx):
        loss, pred, obs = self._shared_step(batch, batch_idx, stage='train')
        self.log("train_loss", loss, batch_size=self.cfg.machine.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred, obs = self._shared_step(batch, batch_idx, stage='val')
        metrics = self.metrics(pred, obs)
        self.log_dict(metrics, batch_size=self.cfg.machine.batch_size)
        self.log("val_loss", loss, batch_size=self.cfg.machine.batch_size)

    def test_step(self, batch, batch_idx):
        loss, pred, obs = self._shared_step(batch, batch_idx, stage='test')
        metrics = self.metrics(pred, obs)
        self.log_dict(metrics, batch_size=self.cfg.machine.batch_size)
        self.log("test_loss", loss, batch_size=self.cfg.machine.batch_size)
        return pred, obs

    def predict_step(self, batch, batch_idx, *args, **kwargs):
        if self.cfg.task.test_mode == 'perturb':
            return self.perturb_step(batch, batch_idx)
        elif self.cfg.task.test_mode == 'predict':
            loss, pred, obs = self._shared_step(
                batch, batch_idx, stage='predict')
            return pred, obs
        elif self.cfg.task.test_mode == 'interpret':
            # assume focus is the center peaks in the input sample
            with torch.enable_grad():
                return self.interpret_step(batch, batch_idx, layer_names=self.cfg.task.layer_names, focus=self.cfg.dataset.n_peaks_upper_bound//2)
        elif self.cfg.task.test_mode == 'perturb_interpret':
            with torch.enable_grad():
                return self.perturb_interpret_step(batch, batch_idx)

    def perturb_step(self, batch, batch_idx):
        """Perturb the input sequence and do inference on both.
        """
        batch_wt = batch['WT']
        batch_mut = batch['MUT']

        input_wt = self.model.get_input(batch_wt, perturb=True)
        output_wt = self(input_wt)
        input_mut = self.model.get_input(batch_mut, perturb=True)
        output_mut = self(input_mut)
        pred_wt, obs_wt = self.model.before_loss(output_wt, batch_wt)
        pred_mut, obs_mut = self.model.before_loss(output_mut, batch_mut)
        return {'pred_wt': pred_wt,
                'obs_wt': obs_wt,
                'pred_mut': pred_mut,
                'obs_mut': obs_mut}

    def perturb_interpret_step(self, batch, batch_idx):
        """Perturb the input sequence and do interpretation on both."""
        batch_wt = batch['WT']
        batch_mut = batch['MUT']

        pred_wt, obs_wt, jacobians_wt, embeddings_wt = self.interpret_step(
            batch_wt, batch_idx, layer_names=self.cfg.task.layer_names, focus=self.cfg.dataset.n_peaks_upper_bound//2)
        pred_mut, obs_mut, jacobians_mut, embeddings_mut = self.interpret_step(
            batch_mut, batch_idx, layer_names=self.cfg.task.layer_names, focus=self.cfg.dataset.n_peaks_upper_bound//2)
        return {'pred_wt': pred_wt,
                'obs_wt': obs_wt,
                'pred_mut': pred_mut,
                'obs_mut': obs_mut,
                'jacobians_wt': jacobians_wt,
                'embeddings_wt': embeddings_wt,
                'jacobians_mut': jacobians_mut,
                'embeddings_mut': embeddings_mut,
                }

    def interpret_step(self, batch, batch_idx, layer_names: List[str] = None, focus: int = None):
        target_tensors = {}
        hooks = []
        input = self.model.get_input(batch)
        assert focus is not None, "Please provide a focus position for interpretation"
        assert layer_names is not None, "Please provide a list of layer names for interpretation"
        for layer_name in layer_names:
            assert layer_name in self.model.get_layer_names(
            ), f"{layer_name} is not valid, valid layers are: {self.model.get_layer_names()}"

        # Register hooks to capture the target tensors
        def capture_target_tensor(name):
            def hook(module, input, output):
                target_tensors[name] = output
                # Retain the gradient of the target tensor
                output.retain_grad()
            return hook

        if layer_names is None or len(layer_names) == 0:
            target_tensors['input'] = input
        else:
            for name in layer_names:
                layer = self.model.get_layer(name)
                hook = layer.register_forward_hook(capture_target_tensor(name))
                hooks.append(hook)

        # Forward pass
        output = self(input)
        pred, obs = self.model.before_loss(output, batch)
        # Remove the hooks after the forward pass
        # for hook in hooks:
        #     hook.remove()
        # Compute the jacobian of the output with respect to the target tensor
        jacobians = {}
        for target_name, target in obs.items():
            jacobians[target_name] = {}
            for i in range(target.shape[-1]):
                output = self(input)
                pred, obs = self.model.before_loss(output, batch)
                jacobians[target_name][str(i)] = {}
                mask = torch.zeros_like(target).to(self.device)
                mask[:, focus, i] = 1
                pred[target_name].backward(mask)
                for name, embed in target_tensors.items():
                    jacobians[target_name][str(
                        i)][name] = embed.grad.detach().cpu().numpy()
                    embed.grad.zero_()

        embeddings = {name: embed.detach().cpu().numpy()
                      for name, embed in target_tensors.items()}
        for target_name, target in pred.items():
            pred[target_name] = target.detach().cpu().numpy()
        for target_name, target in obs.items():
            obs[target_name] = target.detach().cpu().numpy()
        return pred, obs, jacobians, embeddings

    def configure_optimizers(self):
        # a adam optimizer with a scheduler using lightning function
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr)
        return optimizer

    # def configure_optimizers(self):
    #     optimizer = create_optimizer(self.cfg.optimizer, self.model)
    #     num_training_steps_per_epoch = (
    #         self.cfg.dataset.dataset_size // self.cfg.machine.batch_size // self.cfg.machine.num_devices
    #     )
    #     schedule = cosine_scheduler(
    #         base_value=self.cfg.optimizer.lr,
    #         final_value=self.cfg.optimizer.min_lr,
    #         epochs=self.cfg.training.epochs,
    #         niter_per_ep=num_training_steps_per_epoch,
    #         warmup_epochs=self.cfg.training.warmup_epochs,
    #         start_warmup_value=0,
    #         warmup_steps=-1
    #     )
    #     lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    #         optimizer,
    #         lr_lambda=lambda step: schedule[step],
    #     )
    #     return [optimizer], [lr_scheduler]

    def on_validation_epoch_end(self):
        if self.cfg.dataset_name != 'bias_thp1':
            trainer = self.trainer
            metric = run_ppif_task(trainer, self)
            step = trainer.global_step
            self.logger.log_metrics(metric, step)

    # def on_validation_end(self):
    #     # Perform inference on the mutations
    #     predictions = []
    #     for batch in self.inference_dataset:
    #         batch_predictions = self.model(batch)
    #         predictions.append(batch_predictions)

    #     # Run evaluation tasks
    #     for name, task in self.evaluation_tasks.items():
    #         task.predict(predictions)
    #         # log correlation


class GETDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def _shared_build_dataset(self, is_train, sequence_obj=None):
        # config = self.cfg.dataset.dataset_configs[self.cfg.dataset_name]
        # merge config with self.cfg.dataset
        config = DictConfig({**self.cfg.dataset})

        root = self.cfg.machine.data_path
        codebase = self.cfg.machine.codebase
        assembly = self.cfg.assembly
        config.dataset_size = config.dataset_size if is_train else config.eval_dataset_size
        config.zarr_dirs = [
            f'{root}/{zarr_dir}' for zarr_dir in config.zarr_dirs]
        config.genome_seq_zarr = f'{root}/{assembly}.zarr'
        config.genome_motif_zarr = f'{root}/{assembly}_motif_result.zarr'
        config.insulation_paths = [
            f'{codebase}/data/{assembly}_4DN_average_insulation.ctcf.adjecent.feather',
            f'{codebase}/data/{assembly}_4DN_average_insulation.ctcf.longrange.feather']
        self.dataset_config = config
        sequence_obj = sequence_obj if sequence_obj is not None else get_sequence_obj(
            config.genome_seq_zarr)
        gencode_obj = get_gencode_obj(config.genome_seq_zarr, root)

        return config, sequence_obj, gencode_obj

    def build_training_dataset(self, sequence_obj, is_train=True) -> None:
        config, sequence_obj, gencode_obj = self._shared_build_dataset(
            is_train, sequence_obj=sequence_obj)

        # Create dataset with configuration parameters
        dataset = PretrainDataset(
            is_train=is_train,
            sequence_obj=sequence_obj,
            **config
        )

        return dataset

    def build_inference_dataset(self, sequence_obj, gene_list=None):
        config, sequence_obj, gencode_obj = self._shared_build_dataset(
            is_train=False, sequence_obj=sequence_obj)
        if self.mutations is not None:
            # remove from config
            config.pop('mutations')
        # no need to leave out chromosomes or celltypes in inference
        config['leave_out_chromosomes'] = None
        config['random_shift_peak'] = None
        dataset = InferenceDataset(
            is_train=False,
            assembly=self.cfg.assembly,
            gencode_obj=gencode_obj,
            gene_list=gene_list,
            mutations=self.mutations,
            sequence_obj=sequence_obj,
            **config
        )
        return dataset

    def build_perturb_dataset(self, perturbations, perturb_mode, sequence_obj, gene_list=None):
        inference_dataset = self.build_inference_dataset(
            sequence_obj, gene_list=gene_list)
        dataset = PerturbationInferenceDataset(
            inference_dataset, perturbations, perturb_mode)
        return dataset

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        genome_seq_zarr = f'{self.cfg.machine.data_path}/{self.cfg.assembly}.zarr'
        sequence_obj = get_sequence_obj(genome_seq_zarr)

        if stage == 'fit' or stage is None:
            self.dataset_train = self.build_training_dataset(
                sequence_obj=sequence_obj, is_train=True)
            self.dataset_val = self.build_training_dataset(
                sequence_obj=sequence_obj, is_train=False)
        if stage == 'predict':
            if self.cfg.task.test_mode == 'predict':
                self.dataset_predict = self.build_training_dataset(
                    sequence_obj=sequence_obj, is_train=False)
            elif self.cfg.task.test_mode == 'interpret':
                self.dataset_predict = self.build_inference_dataset(
                    sequence_obj=sequence_obj)
            elif 'perturb' in self.cfg.task.test_mode:
                self.mutations = pd.read_csv(self.cfg.task.mutations, sep='\t')
                if self.mutations is not None:
                    self.perturbation_mode = 'mutation'
                elif self.cfg.dataset.peak_inactivation is not None:
                    self.perturbation_mode = 'peak_inactivation'
                self.dataset_predict = self.build_perturb_dataset(
                    perturbations=self.mutations, perturb_mode=self.perturbation_mode,
                    sequence_obj=sequence_obj, gene_list=self.cfg.task.gene_list)

        if stage == 'validate':
            self.dataset_val = self.build_training_dataset(
                sequence_obj=sequence_obj, is_train=False)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.cfg.machine.batch_size,
            num_workers=self.cfg.machine.num_workers,
            drop_last=True,
            collate_fn=get_rev_collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_val,
            batch_size=self.cfg.machine.batch_size,
            num_workers=self.cfg.machine.num_workers,
            drop_last=True,
            collate_fn=get_rev_collate_fn,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_test,
            batch_size=self.cfg.machine.batch_size,
            num_workers=self.cfg.machine.num_workers,
            drop_last=True,
            collate_fn=get_rev_collate_fn if 'perturb' not in self.cfg.task.test_mode else get_perturb_collate_fn,
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_predict,
            batch_size=self.cfg.machine.batch_size,
            num_workers=self.cfg.machine.num_workers,
            drop_last=False,
            collate_fn=get_rev_collate_fn if 'perturb' not in self.cfg.task.test_mode else get_perturb_collate_fn,
        )


def run(cfg: DictConfig):
    torch.set_float32_matmul_precision('medium')
    model = LitModel(cfg)
    dm = GETDataModule(cfg)
    model.dm = dm
    wandb = WandbLogger(name=cfg.wandb.run_name,
                        project=cfg.wandb.project_name)
    wandb.log_hyperparams(dict(cfg))
    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="gpu",
        num_sanity_val_steps=10,
        strategy="auto",
        devices=cfg.machine.num_devices,
        # callbacks=[ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, save_last=True, filename="best"),
        #            LearningRateMonitor(logging_interval='epoch')],
        logger=[wandb,
                CSVLogger('logs', f'{cfg.wandb.project_name}_{cfg.wandb.run_name}')],
        callbacks=[ModelCheckpoint(
            monitor="val_loss", mode="min", save_top_k=1, save_last=True, filename="best")],
        plugins=[MixedPrecision(precision='16-mixed', device="cuda")],
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        gradient_clip_val=cfg.training.clip_grad,
        log_every_n_steps=25,
        default_root_dir=cfg.machine.output_dir,
    )
    # tuner = Tuner(trainer)
    # tuner.lr_find(model, datamodule=dm)
    if cfg.stage == 'fit':
        trainer.fit(model, dm, ckpt_path=cfg.finetune.checkpoint)
    if cfg.stage == 'validate':
        trainer.validate(model, datamodule=dm)
    if cfg.stage == 'predict':
        trainer.predict(model, datamodule=dm)


def run_downstream(cfg: DictConfig):
    torch.set_float32_matmul_precision('medium')
    # if cfg.finetune.checkpoint is not None:
    # model = LitModel.load_from_checkpoint(cfg.finetune.checkpoint)
    # else:
    model = LitModel(cfg)
    # move the model to the gpu
    model.to('cuda')
    dm = GETDataModule(cfg)
    model.dm = dm
    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="gpu",
        num_sanity_val_steps=10,
        strategy="auto",
        devices=cfg.machine.num_devices,
        # plugins=[MixedPrecision(precision='16-mixed', device="cuda")],
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        gradient_clip_val=cfg.training.clip_grad,
        log_every_n_steps=100,
        deterministic=True,
        default_root_dir=cfg.machine.output_dir,
    )
    print(run_ppif_task(trainer, model))


def run_ppif_task(trainer: L.Trainer, lm: LitModel, output_key='atpm'):
    import numpy as np
    from scipy.stats import pearsonr, spearmanr
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    mutation = pd.read_csv(
        lm.cfg.task.mutations, sep='\t')
    n_mutation = mutation.shape[0]
    n_peaks_upper_bound = lm.cfg.dataset.n_peaks_upper_bound
    result = []
    # setup dataset_predict
    lm.dm.setup(stage='predict')
    with torch.no_grad():
        lm.to("cuda")
        for i, batch in tqdm(enumerate(lm.dm.predict_dataloader()), total=len(lm.dm.predict_dataloader())):
            batch = lm.transfer_batch_to_device(
                batch, lm.device, dataloader_idx=0)
            out = lm.predict_step(batch, i)
            result.append(out)
    pred_wt = [r['pred_wt'][output_key] for r in result]
    pred_mut = [r['pred_mut'][output_key] for r in result]
    n_celltypes = lm.dm.dataset_predict.inference_dataset.datapool.n_celltypes
    pred_wt = torch.cat(pred_wt, dim=0).reshape(
        n_celltypes, n_mutation, n_peaks_upper_bound)[0, :, n_peaks_upper_bound//2]
    pred_mut = torch.cat(pred_mut, dim=0).reshape(
        n_celltypes, n_mutation, n_peaks_upper_bound)[0, :, n_peaks_upper_bound//2]
    pred_change = (10**pred_mut - 10**pred_wt) / \
        (10**pred_wt - 1) * 100
    mutation['pred_change'] = pred_change.detach().cpu().numpy()
    y = mutation.query('`corrected p value`<=0.05').query('Screen.str.contains("Pro")').query('Screen.str.contains("Tiling")')[
        '% change to PPIF expression'].values
    x = mutation.query('`corrected p value`<=0.05').query('Screen.str.contains("Pro")').query(
        'Screen.str.contains("Tiling")')['pred_change'].values
    pearson = np.corrcoef(x, y)[0, 1]
    r2 = r2_score(y, x)
    spearman = spearmanr(x, y)[0]
    slope = LinearRegression().fit(x.reshape(-1, 1), y).coef_[0]
    # save a scatterplot
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.scatterplot(x=x, y=y)
    plt.xlabel('Predicted change in PPIF expression')
    plt.ylabel('Observed change in PPIF expression')
    plt.savefig(
        f'{lm.cfg.machine.output_dir}/ppif_scatterplot.png')
    return {
        'ppif_pearson': pearson,
        'ppif_spearman': spearman,
        'ppif_r2': r2,
        'ppif_slope': slope
    }
