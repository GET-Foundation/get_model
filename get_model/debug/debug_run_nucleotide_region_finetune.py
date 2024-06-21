import hydra
import sys
import gc
import torch
from get_model.run_everything import run

from get_model.config.config import *


@hydra.main(config_path="../config", config_name="eval_k562_fetal_nucleotide_region_finetune_hic", version_base="1.3")
def main(cfg: Config):
    run(cfg)


if __name__ == "__main__":
    main()
    gc.collect()
    torch.cuda.empty_cache()
    sys.exit()
