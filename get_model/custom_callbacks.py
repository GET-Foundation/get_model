import os
from lightning.pytorch.callbacks import ModelCheckpoint

class FinetunedModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _save_checkpoint(self, trainer, filepath):
        # Get the original checkpoint
        checkpoint = trainer.checkpoint_connector.dump_checkpoint()

        # Filter out parameters with 'original' in the key name
        finetuned_state_dict = {k: v for k, v in checkpoint['state_dict'].items() if 'original' not in k}
        checkpoint['state_dict'] = finetuned_state_dict

        # Save the modified checkpoint
        trainer.strategy.save_checkpoint(checkpoint, filepath)

    def on_train_end(self, trainer, pl_module):
        if self.save_last:
            # Save finetuned version of 'last' checkpoint
            filepath = os.path.join(self.dirpath, "last-finetuned.ckpt")
            self._save_checkpoint(trainer, filepath)

        if self.save_top_k != 0:
            # Save finetuned version of 'best' checkpoint
            filepath = os.path.join(self.dirpath, "best-finetuned.ckpt")
            self._save_checkpoint(trainer, filepath)