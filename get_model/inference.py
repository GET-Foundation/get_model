import hydra
from get_model.config.config import Config
from hydra.core.global_hydra import GlobalHydra

def load_config(config_name):
    # Initialize Hydra to load the configuration
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="../get_model/config", version_base="1.3")
    cfg = hydra.compose(config_name=config_name)
    return cfg

def setup_model_from_config(cfg: Config, type='everything'):
    # Depending on the stage, create the appropriate DataModule and DataLoader
    from get_model.run_everything import RegionLitModel as EverythingLitModel
    from get_model.run_ref_region import RegionLitModel as ReferenceRegionLitModel
    from get_model.run_region import RegionLitModel
    from get_model.run import LitModel
    if type == 'everything':
        m = EverythingLitModel(cfg)
    elif type == 'ref_region':
        m = ReferenceRegionLitModel(cfg)
    elif type == 'region':
        m = RegionLitModel(cfg)
    elif type == 'nucleotide':
        m = LitModel(cfg)

    return m


def main(args):
    cfg = load_config(args.config_name)
    m = setup_model_from_config(cfg, args.type)
    return m

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-name', type=str, default='config')
    parser.add_argument('--type', type=str, default='everything')
    args = parser.parse_args()
    main(args)