from hydra import initialize, compose
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra
from get_model.model.model_refactored import *
def clear_hydra():
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

def _model_forward(model_name):
    clear_hydra()  # Clear Hydra's global state before initialization
    with initialize(config_path="../get_model/config/model", version_base='1.3'):
        cfg = compose(config_name=model_name)

        model = instantiate(cfg.model)
        assert model.test() is not None, "Model test method should return a non-None result"
        assert model.__class__.__name__ == model_name, f"Model class name should be {model_name}"


def test_GETPretrain():
    _model_forward("GETPretrain")
def test_GETPretrainMaxNorm():
    _model_forward("GETPretrainMaxNorm")
def test_GETFinetune():
    _model_forward("GETFinetune")
def test_GETFinetuneMaxNorm():
    _model_forward("GETFinetuneMaxNorm")
def test_GETChrombpNetBias():
    _model_forward("GETChrombpNetBias")
def test_GETChrombpNet():
    _model_forward("GETChrombpNet")