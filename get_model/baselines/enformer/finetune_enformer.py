import argparse
import datetime
import json
import os
import time
from collections import OrderedDict
from pathlib import Path
import polars as pl
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from timm.data.mixup import Mixup
from timm.models import create_model
from timm.utils import ModelEma
import wandb

from enformer_pytorch import from_pretrained, Enformer, GenomeIntervalDataset
from enformer_pytorch.finetune import HeadAdapterWrapper
from enformer_pytorch.data import FastaInterval, identity

import get_model.utils as utils
from get_model.utils import NativeScalerWithGradNormCount as NativeScaler
from get_model.baselines.enformer.engine import evaluate_all
from get_model.baselines.enformer.engine import finetune_train_one_epoch as train_one_epoch
from get_model.optim import (
    LayerDecayValueAssigner,
    create_optimizer,
    get_parameter_groups,
)


torch.autograd.set_detect_anomaly(True)

hg38_path = "/pmglocal/alb2281/get_data/get_resources/hg38.ml.fa"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GenomeIntervalFinetuneDataset(Dataset):
    def __init__(
        self,
        bed_file,
        fasta_file,
        filter_df_fn = identity,
        chr_bed_to_fasta_map = dict(),
        context_length = None,
        return_seq_indices = False,
        shift_augs = None,
        rc_aug = False,
        return_augs = False
    ):
        super().__init__()
        bed_path = Path(bed_file)
        assert bed_path.exists(), 'path to .bed file must exist'

        df = pl.read_csv(str(bed_path), separator = '\t', has_header = False)
        df = filter_df_fn(df)
        self.df = df
        self.chr_bed_to_fasta_map = chr_bed_to_fasta_map
        self.fasta = FastaInterval(
            fasta_file = fasta_file,
            context_length = context_length,
            return_seq_indices = return_seq_indices,
            shift_augs = shift_augs,
            rc_aug = rc_aug
        )

        self.return_augs = return_augs

    def __len__(self):
        return len(self.df)

    # Modify to return with target
    def __getitem__(self, ind):
        interval = self.df.row(ind)
        chr_name, start, end = (interval[0], interval[1], interval[2])
        chr_name = self.chr_bed_to_fasta_map.get(chr_name, chr_name)
        return self.fasta(chr_name, start, end, return_augs = self.return_augs), interval[3]


def split_data_by_chr(base_dir, leaveout_chr):    
    bed_df = pd.read_csv(atac_data, index_col="index")
    labels = np.array(load_npz(labels_path).todense()[:,-1].reshape(-1))[0]
    bed_df["target"] = labels
    train_df = bed_df[bed_df["Chromosome"] != leaveout_chr]
    val_df = bed_df[bed_df["Chromosome"] == leaveout_chr]
    train_path = f"{base_dir}/train.bed"
    val_path = f"{base_dir}/val.bed"
    train_df.to_csv(train_path, sep="\t", header=False, index=False)
    val_df.to_csv(val_path, sep="\t", header=False, index=False)
    return train_path, val_path


def train_enformer(args):
    utils.init_distributed_mode(args)

    print(args)

    if utils.is_main_process(): # Log metrics only on main process
        wandb.login()
        run = wandb.init(
            project=args.wandb_project_name,
            name=args.wandb_run_name,
        )

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    base_dir = f"/pmglocal/alb2281/get_data/k562_count_10/splits/{args.leave_out_chromosomes}"

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        train_path, val_path = split_data_by_chr(base_dir, args.leave_out_chromosomes)
    else:
        train_path = f"{base_dir}/train.bed"
        val_path = f"{base_dir}/val.bed"
    
    dataset_train = GenomeIntervalFinetuneDataset(
        bed_file = train_path,                         
        fasta_file = hg38_path,                   
        return_seq_indices = True,                          
        shift_augs = (-2, 2),                              
        context_length = 196_608,
    )
    dataset_val = GenomeIntervalFinetuneDataset(
        bed_file = val_path,                         
        fasta_file = hg38_path,                   
        return_seq_indices = True,                          
        shift_augs = (-2, 2),                              
        context_length = 196_608,
    )

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
            )
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)


    log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
    else:
        data_loader_val = None

    mixup_fn = None
    enformer = from_pretrained('EleutherAI/enformer-official-rough', target_length=2) # target_length=1 throws error
    model = HeadAdapterWrapper(
        enformer = enformer,
        num_tracks = 1,
        post_transformer_embed = False,
        auto_set_target_length = False, # Override infer target_length from target dimensions
    ).to(device)
    model.to(device)
    model_ema = None
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print("number of params:", n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    args.lr = args.lr * total_batch_size / 256
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    if dataset_val is not None:
        print("Number of testing examples = %d" % len(dataset_val))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu],
            find_unused_parameters=True,
        )
        model_without_ddp = model.module
    print("Complete DistributedDataParallel!")

    optimizer = create_optimizer(
        args,
        model_without_ddp,
        skip_list=None,
        get_num_layer=None,
        get_layer_scale=None,
    )
    loss_scaler = NativeScaler()

    print("Use step level LR scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        num_training_steps_per_epoch,
    )
    print(
        "Max WD = %.7f, Min WD = %.7f"
        % (max(wd_schedule_values), min(wd_schedule_values))
    )

    utils.auto_load_model(
        args=args,
        model=model,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        model_ema=model_ema,
    )

    if args.eval:
        test_stats = evaluate_all(data_loader_val, model, device, criterion, args)
        max_r2score = test_stats["r2score"]
        max_pearsonr_score = test_stats["pearsonr_score"]
        max_spearmanr_score = test_stats["spearmanr_score"]

        print(f"Statistics of the network on the {len(dataset_val)} test images")
        print(
            f"R2score: {max_r2score:.3f}, pearsonr_score: {max_pearsonr_score:.3f}, spearmanr_score: {max_spearmanr_score:.3f}"
        )
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_r2score = 0.0
    max_pearsonr_score = 0.0
    max_spearmanr_score = 0.0
    max_r2score_atac = 0.0
    max_pearsonr_score_atac = 0.0
    max_spearmanr_score_atac = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats = train_one_epoch(
            model,
            data_loader_train,
            data_loader_val,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            model_ema,
            mixup_fn,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            update_freq=args.update_freq,
            args=args,
        )
        print("*******Epoch done********")
        if args.output_dir and args.save_ckpt:
            print("*******Saving models********")
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                    model_ema=model_ema,
                )
        
        if (
            data_loader_val is not None
            and args.eval_freq > 0
            and (epoch + 1) % args.eval_freq == 0
        ):
            print("*******Start evaluation********")
            test_stats = evaluate_all(
                data_loader_val, model, device, criterion, args, epoch, printlog=True
            )
            print(
                f"R2, Pearson, Spearmanr Score of the network on the {len(dataset_val)} test expression: {test_stats['r2score']:.1f}, {test_stats['pearsonr_score']:.1f}, {test_stats['spearmanr_score']:.1f}"
            )
            if max_r2score < test_stats["r2score"]:
                max_r2score = test_stats["r2score"]
                max_pearsonr_score = test_stats["pearsonr_score"]
                max_spearmanr_score = test_stats["spearmanr_score"]

                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args,
                        model=model,
                        model_without_ddp=model_without_ddp,
                        optimizer=optimizer,
                        loss_scaler=loss_scaler,
                        epoch="best",
                        model_ema=model_ema,
                    )

            print(
                f"Max r2score: {max_r2score:.3f}, Max pearsonr_score: {max_pearsonr_score:.3f}, Max spearmanr_score: {max_spearmanr_score:.3f}"
            )
            if log_writer is not None:
                log_writer.update(
                    test_r2score=test_stats["r2score"], head="perf", step=epoch
                )
                log_writer.update(
                    test_spearmanr_score=test_stats["pearsonr_score"],
                    head="perf",
                    step=epoch,
                )
                log_writer.update(
                    test_spearmanr_score=test_stats["spearmanr_score"],
                    head="perf",
                    step=epoch,
                )
                log_writer.update(test_loss=test_stats["loss"], head="perf", step=epoch)

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }
        else:
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }
        
        if utils.is_main_process():
            wandb.log(log_stats)
        
        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--update_freq", default=1, type=int)
    parser.add_argument("--save_ckpt_freq", default=1, type=int)
    # wandb params
    parser.add_argument("--wandb_project_name", type=str, default="get-finetune", help="wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name")
    # torch compile
    parser.add_argument(
        "--compile_model",
        action="store_true",
        help="compile model with torch compile",
    )

    # Optimizer parameters
    parser.add_argument(
        "--opt",
        default="adamw",
        type=str,
        metavar="OPTIMIZER",
        help='Optimizer (default: "adamw"',
    )
    parser.add_argument(
        "--opt_eps",
        default=1e-8,
        type=float,
        metavar="EPSILON",
        help="Optimizer Epsilon (default: 1e-8)",
    )
    parser.add_argument(
        "--opt_betas",
        default=None,
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: None, use opt default)",
    )
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )
    parser.add_argument(
        "--weight_decay_end",
        type=float,
        default=None,
        help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="learning rate (default: 1e-3)",
    )
    parser.add_argument("--layer_decay", type=float, default=0.75)

    parser.add_argument(
        "--warmup_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="warmup learning rate (default: 1e-6)",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-5,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0 (1e-5)",
    )

    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=1,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=-1,
        metavar="N",
        help="num of steps to warmup LR, will overload warmup_epochs if set > 0",
    )
    parser.add_argument(
        "--mask_ratio",
        default=0.5,
        type=float,
        help="ratio of the visual tokens/patches need be masked",
    )

    # Dataset parameters
    parser.add_argument(
        "--atac_data",
        default=None,
        type=str,
        help="atac dataset path",
    )
    parser.add_argument(
        "--labels_path", default=None, type=str, help="labels path"
    )

    parser.add_argument(
        "--output_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--no_auto_resume", action="store_false", dest="auto_resume")
    parser.set_defaults(auto_resume=True)

    parser.add_argument("--save_ckpt", action="store_true")
    parser.add_argument("--no_save_ckpt", action="store_false", dest="save_ckpt")
    parser.set_defaults(save_ckpt=True)

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--eval_freq",
        default=1,
        type=int,
        metavar="N",
    )
    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation",
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument("--distributed", default=True, action="store_true")
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    parser.add_argument("--eval_each_step", action="store_true", default=False)
    parser.add_argument("--eval_tss", action="store_true", default=False)
    parser.add_argument("--target_type", default="Log", type=str)
    parser.add_argument("--target_thresh", default=32, type=int)

    parser.add_argument("--leave_out_chromosomes", default="chr4", type=str)
    args = parser.parse_args()
    train_enformer(args)
