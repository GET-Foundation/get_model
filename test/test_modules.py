#%%
import hydra

from get_model.model.model_refactored import GETPretrain, GETPretrainConfig

hydra.initialize(config_path="../get_model/config/model/pretrain")
cfg = hydra.compose(config_name="template")
#%%
model = GETPretrain(cfg.model.cfg)
model.test()
# %%
    