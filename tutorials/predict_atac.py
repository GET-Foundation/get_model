#%%
from get_model.config.config import load_config, pretty_print_config

from get_model.run_region import run_zarr as run
#%%
%matplotlib inline

cfg = load_config('finetune_tutorial') # load the predefined finetune tutorial config; it's in get_model/config/finetune_tutorial.yaml
cfg.run.run_name='predict_atac' # this is a unique name for this run
cfg.dataset.zarr_path = "./output.zarr" # the tutorial data which contains astrocyte atac & rna
cfg.dataset.celltypes = 'astrocyte' # the celltypes you want to finetune
cfg.dataset.leave_out_chromosomes = 'chr10,chr11'
cfg.run.use_wandb=True # this is a logging system, you can turn it off by setting it to False
cfg.machine.num_devices=1 # this is the number of GPUs you want to use
cfg.machine.batch_size = 32 # this is the batch size you want to use, checkout `nvidia-smi` to see how many GPUs you have and how much memory is available
cfg.training.epochs = 10 # this is the number of epochs you want to train for
# %%
# Switch model to finetune ATAC model, this model will set all ATAC values for each peak to 1, and use motif information to predict the ATAC values
cfg.model = load_config('model/GETRegionFinetuneATAC').model.model
pretty_print_config(cfg)
# %%
# first run the model without initializing with a pretrain checkpoint
cfg.finetune.checkpoint = None
trainer = run(cfg)
# %%
trainer.callback_metrics
# %%
# now train the model with a pretrain checkpoint
#  download from https://us-east-1.console.aws.amazon.com/s3/object/2023-get-xf2217?region=us-east-1&bucketType=general&prefix=get_demo/checkpoints/regulatory_inference_checkpoint_fetal_adult/pretrain_fetal_adult/checkpoint-799.pth
#! aws s3 cp s3://2023-get-xf2217/get_demo/checkpoints/regulatory_inference_checkpoint_fetal_adult/pretrain_fetal_adult/checkpoint-799.pth ./checkpoint-799.pth
cfg.finetune.checkpoint = './checkpoint-799.pth'
cfg.run.run_name = 'predict_atac_with_pretrain'
cfg.finetune.strict = False # need this because the original checkpoint has a mask-prediction header
trainer = run(cfg)
# %%
trainer.callback_metrics
# %%
# Seems pretraining checkpoint deos help with the convergence rate; altough both model is not undertrained
# %%
