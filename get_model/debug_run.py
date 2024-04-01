import hydra
from run import run, run_downstream

from get_model.config.config import *


@hydra.main(config_path="config", config_name="finetune_k562", version_base="1.3")
def main(cfg: Config):
    run_downstream(cfg)


if __name__ == "__main__":
    main()
