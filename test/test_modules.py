from hydra import initialize, compose
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra
from get_model.model.model_refactored import *

def clear_hydra():
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

def test_GETPretrain():
    clear_hydra()  # Clear Hydra's global state before initialization
    with initialize(config_path="../get_model/config/model", version_base='1.3'):
        cfg = compose(config_name="GETPretrain")

        model = instantiate(cfg.model)
        assert model.test() is not None, "Model test method should return a non-None result"
        assert model.__class__.__name__ == "GETPretrain", "Model class name should be GETPretrain"

def test_GETPretrainMaxNorm():
    clear_hydra()  # Clear Hydra's global state before initialization
    with initialize(config_path="../get_model/config/model", version_base='1.3'):
        cfg = compose(config_name="GETPretrainMaxNorm")

        model = instantiate(cfg.model)
        assert model.test() is not None, "Model test method should return a non-None result"
        assert model.__class__.__name__ == "GETPretrainMaxNorm", "Model class name should be GETPretrainMaxNorm"


def test_GETFinetune():
    clear_hydra()  # Clear Hydra's global state before initialization
    with initialize(config_path="../get_model/config/model", version_base='1.3'):
        cfg = compose(config_name="GETFinetune")

        model = instantiate(cfg.model)
        assert model.test() is not None, "Model test method should return a non-None result"
        assert model.__class__.__name__ == "GETFinetune", "Model class name should be GETFinetune"

def test_GETFinetuneMaxNorm():
    clear_hydra()  # Clear Hydra's global state before initialization
    with initialize(config_path="../get_model/config/model", version_base='1.3'):
        cfg = compose(config_name="GETFinetuneMaxNorm")

        model = instantiate(cfg.model)
        assert model.test() is not None, "Model test method should return a non-None result"
        assert model.__class__.__name__ == "GETFinetuneMaxNorm", "Model class name should be GETFinetuneMaxNorm"