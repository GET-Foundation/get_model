import hydra
from run import run, run_downstream

from get_model.config.config import *


@hydra.main(config_path="config", config_name="finetune_k562", version_base="1.3")
def run_k562(cfg: Config):
    run_downstream(cfg)


@hydra.main(config_path="config", config_name="finetune_gbm", version_base="1.3")
def run_gbm(cfg: Config):
    run_downstream(cfg)


if __name__ == "__main__":
    run_gbm()
