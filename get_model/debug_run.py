import hydra
from run import run

from get_model.config.config import *


@hydra.main(config_path="config", config_name="chrombpnet_k562", version_base="1.3")
def main(cfg: Config):
    run(cfg)


if __name__ == "__main__":
    main()
