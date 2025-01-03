import hydra

from get_model.config.config import Config
from get_model.run_region import run


@hydra.main(config_path="../config", config_name="fetal_region", version_base="1.3")
def main(cfg: Config):
    run(cfg)


if __name__ == "__main__":
    main()
