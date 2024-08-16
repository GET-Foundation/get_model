import hydra
import torch
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


class InferenceModel():
    def __init__(self, checkpoint_path, device='cuda', config_name='eval_gbm_fetal_ref_region', type='ref_region'):
        """Compatible with the old mutation code"""
        self.config_name = config_name
        self.type = type
        if checkpoint_path is not None and 'state_dict' in checkpoint_path:
            self.model = self.model.load_from_checkpoint(checkpoint_path)
        elif checkpoint_path is not None and 'model' in checkpoint_path:
            # old checkpoint
            state_dict = torch.load(checkpoint_path)['model']
            self.cfg = load_config(config_name)
            self.model = setup_model_from_config(self.cfg, type)
            self.model.load_state_dict(state_dict)
        else:
            raise('No checkpoint provided')
        self.model.to(device)

    def predict(self, region_motif):
        batch = {'region_motif': region_motif,
                 'exp_label': 0}
        loss, pred, obs = self.model_shared_step(
                batch, None, stage='predict')
        return pred.cpu().numpy()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-name', type=str, default='config')
    parser.add_argument('--type', type=str, default='everything')
    args = parser.parse_args()
    main(args)