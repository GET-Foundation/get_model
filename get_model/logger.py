from omegaconf import DictConfig
import hydra
import os
import pandas as pd

class BaseLogger:
    def __init__(self, config: DictConfig):
        self.config = config
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.config})"

    def update(self, stat_dict, step=None, log_freq=10):
        raise NotImplementedError("Subclasses must implement this method.")

    def flush(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def log_image(self, image, name):
        raise NotImplementedError("Subclasses must implement this method.")

    def wrap_up(self):
        raise NotImplementedError("Subclasses must implement this method.")

class TensorboardLogger(BaseLogger):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        from tensorboardX import SummaryWriter
        self.writer = SummaryWriter(logdir=self.config.log_dir)
        self.step = 0
        

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, stat_dict, step=None):
        step = step if step is not None else self.step
        for section, metrics in stat_dict.items():
            self.writer.add_scalars(
                section, metrics, global_step=step if step is not None else self.step)
            
    def flush(self):
        self.writer.flush()

    def log_image(self, image, name):
        self.writer.add_image(name, image, global_step=self.step)

    def wrap_up(self):
        self.writer.close()


class WandBLogger(BaseLogger):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        import wandb

        wandb.login()
        run = wandb.init(
            project=self.config.project_name,
            name=self.config.run_name,
        )
        self.run = run

    def update(self, stat_dict, step=None):
        self.run.log(stat_dict, step=step, commit=True)

    def flush(self):
        pass

    def log_image(self, image, name):
        import wandb

        self.run.log({name: wandb.Image(image)})

    def wrap_up(self):
        self.run.finish()

class LocalLogger(BaseLogger):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        os.makedirs(self.config.log_dir, exist_ok=True)
        self.log_dataframe = pd.DataFrame()

    def update(self, stat_dict, step=None):
        filename = os.path.join(self.config.log_dir, f"metrics.csv")
        with open(filename, 'w') as f:
            print(f"Logging to {filename}")
            for section, metrics in stat_dict.items():
                row = {"step": step,
                       "section": section,
                       **metrics}
                self.log_dataframe = pd.concat([self.log_dataframe, pd.DataFrame([row])])

    def flush(self):
        pass

    def log_image(self, image, name):
        image_folder = os.path.join(self.config.log_dir, name)
        os.makedirs(image_folder, exist_ok=True)
        image_path = os.path.join(image_folder, f"{name}_{self.step}.png")
        image.save(image_path)

    def wrap_up(self):
        filename = os.path.join(self.config.log_dir, f"metrics.csv")
        self.log_dataframe.to_csv(filename, index=False)

class Loggers:
    def __init__(self, config: DictConfig):
        self.loggers = []
        self.log_freq = config.log_freq
        for logger_cfg in config.loggers:
            if logger_cfg.type == 'tensorboard':
                self.loggers.append(TensorboardLogger(logger_cfg.tensorboard))
            elif logger_cfg.type == 'wandb':
                self.loggers.append(WandBLogger(logger_cfg.wandb))
            elif logger_cfg.type == 'local':
                self.loggers.append(LocalLogger(logger_cfg.local))
        print(self.__repr__())

    def __repr__(self) -> str:
        return f"Loggers({self.loggers})"
    
    def update(self, stat_dict, step=None):
        if step is not None and step % self.log_freq == 0:
            for logger in self.loggers:
                logger.update(stat_dict, step=step)

    def flush(self):
        for logger in self.loggers:
            logger.flush()

    def log_image(self, image, name):
        for logger in self.loggers:
            logger.log_image(image, name)

    def wrap_up(self):
        for logger in self.loggers:
            logger.wrap_up()
    

@hydra.main(config_path="config", config_name="debug", version_base="1.3")
def main(cfg: DictConfig) -> None:
    loggers = Loggers(cfg)

    # Example usage
    for step in range(20):
        stat_dict = {"train": {"metric": 1.0, "pearson": 0.5},
                    "test": {"metric1": 2.0, "pearson1": 0.6}}
        loggers.update(stat_dict, step=step)
    loggers.flush()
    loggers.wrap_up()


if __name__ == "__main__":
    main()


