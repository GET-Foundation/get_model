import hydra
from get_model.run import run, run_downstream

from get_model.config.config import Config


@hydra.main(config_path="../config", config_name="finetune_k562_fetal", version_base="1.3")
def main(cfg: Config):
    if cfg.stage == "downstream":
        run_downstream(cfg)
    else:
        run(cfg)


if __name__ == "__main__":
    main()
