# %%
import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch

from get_model.config.config import *
from get_model.run import GETDataModule, LitModel


# %%
hydra.initialize_config_dir(
    config_dir="/home/xf2217/Projects/get_model/get_model/config")
# %%
cfg = hydra.compose(config_name="finetune_k562")

# %%
dm = GETDataModule(cfg, mutations=mutation)
# %%
torch.set_float32_matmul_precision('medium')
# %%
model = LitModel(cfg)
# %%
trainer = L.Trainer(
    max_epochs=cfg.training.epochs,
    accelerator="gpu",
    inference_mode=cfg.task.test_mode != "interpret",
    num_sanity_val_steps=10,
    strategy="auto",
    devices=cfg.machine.num_devices,
    accumulate_grad_batches=cfg.training.accumulate_grad_batches,
    gradient_clip_val=cfg.training.clip_grad,
    log_every_n_steps=100,
    deterministic=True,
    default_root_dir=cfg.machine.output_dir,
)

# %%
