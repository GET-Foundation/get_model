import hydra
import sys
import gc
import torch
from get_model.run_ref_region import run

from get_model.config.config import *


@hydra.main(config_path="../config", config_name="eval_k562_fetal_ref_region_k562_hic_abc_lora", version_base="1.3")
def main(cfg: Config):
    run(cfg)


if __name__ == "__main__":
    main()
    gc.collect()
    torch.cuda.empty_cache()
    sys.exit()
