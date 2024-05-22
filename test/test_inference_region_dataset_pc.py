from caesar.io.gencode import Gencode
from hydra import compose, initialize
from get_model.config.config import *
from get_model.run_ref_region import *
from get_model.run_region import RegionDataModule


def test_fetal_region():
    initialize(version_base=None, config_path="../get_model/config")
    cfg = compose(config_name="fetal_region")
    dm = RegionDataModule(cfg)

    gencode_config = {
        "assembly": "hg38",
        "version": 40,
        "gtf_dir": "/home/xf2217/Projects/caesar/data/"
    }
    gencode = Gencode(**gencode_config)

    data = dm.build_inference_dataset(
        is_train=False,
        gene_list="/home/xf2217/Projects/get_revision/brain_multiome/genes_in_Q5.txt",
        gencode_obj=gencode
    )

    # Add assertions to validate the expected behavior
    assert data is not None
