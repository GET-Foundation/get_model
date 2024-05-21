import argparse
import gc
import sys

import hydra
import torch

from get_model.config.config import *
from get_model.run import run as run_v3
from get_model.run_ref_region import run as run_ref_region
from get_model.run_region import run as run_region


def get_meta_config():
    parser = argparse.ArgumentParser(description="Dynamic config name for Hydra")
    parser.add_argument("--setting", type=str, required=True, help="The setting to use", choices=["ref_region", "region", "v3"])
    parser.add_argument("--config_name", type=str, required=True, help="The name of the Hydra config to use")
    args = parser.parse_args()
    return args.setting, args.config_name

setting, config_name = get_meta_config()

@hydra.main(config_path="../config", config_name=config_name, version_base="1.3")
def main(cfg: Config):
    if setting == "ref_region":
        run_ref_region(cfg)
    elif setting == "region":
        run_region(cfg)
    elif setting == "v3":
        run_v3(cfg)
    else:
        raise ValueError(f"Invalid setting: {setting}")


if __name__ == "__main__":
    main()
    gc.collect()
    torch.cuda.empty_cache()
    sys.exit()

