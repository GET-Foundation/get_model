import hydra
from get_model.run_ref_region import run

from get_model.config.config import *


@hydra.main(config_path="../config", config_name="finetune_gbm", version_base="1.3")
def main(cfg: Config):
    run(cfg)


if __name__ == "__main__":
    main()