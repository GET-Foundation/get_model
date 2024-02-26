import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import utils
from dataset.dataset import build_dataset_zarr as build_dataset
from dataset.zarr_dataset import DenseZarrIO, worker_init_fn_get
from engine import evaluate_pretrain
from engine import pretrain_one_epoch as train_one_epoch
from optim import create_optimizer
from timm.models import create_model
from utils import NativeScalerWithGradNormCount as NativeScaler

import get_model.model.model
import wandb
from get_model.dataset.collate import get_rev_collate_fn


def get_args_parser():
    parser = argparse.ArgumentParser("GET pre-training script", add_help=False)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--save_ckpt_freq", default=1, type=int)
    # wandb params
    parser.add_argument("--wandb_project_name", type=str, default="get-pretrain", help="wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name")
    # Model parameters
    parser.add_argument(
        "--model",
        default="get_pretrain_motif",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument(
        "--compile_model",
        action="store_true",
        help="compile model with torch compile",
    )
    # Dataset parameters
    parser.add_argument(
        "--num_region_per_sample",
        default=200,
        type=int,
        help="number of regions for each sample",
    )

    parser.add_argument(
        "--peak_name",
        default="peaks_p0.01",
        type=str,
        help="peak name to use",
    )

    parser.add_argument(
        "--preload_count",
        default=200,
        type=int,
        help="number of samples to preload",
    )

    parser.add_argument(
        "--n_packs",
        default=5,
        type=int,
        help="number of samples per window",
    )

    parser.add_argument(
        "--max_peak_length",
        default=5000,
        type=int,
        help="maximum peak length",
    )

    parser.add_argument(
        "--center_expand_target",
        default=1000,
        type=int,
        help="center expand target",
    )

    parser.add_argument(
        "--n_peaks_lower_bound",
        default=5,
        type=int,
        help="lower bound of number of peaks",
    )

    parser.add_argument(
        "--n_peaks_upper_bound",
        default=200,
        type=int,
        help="upper bound of number of peaks",
    )
    parser.add_argument(
        "--non_redundant",
        default=None,
        choices=["max_depth", "depth_512", "depth_1024", "depth_2048", "depth_4096", None],
    )
    parser.add_argument(
        "--num_motif",
        default=1273,
        type=int,
        help="number of motifs for each region",
    )
    parser.add_argument(
        "--use_insulation",
        action="store_true",
        default=False,
        help="use insulation score",
    )
    parser.add_argument(
        "--invert_peak",
        default=None,
        choices=[0.1, None],
        help="Probability to generate background peaks as null samples"
    )
    parser.add_argument(
        "--random_shift_peak",
        action="store_true",
        default=False,
        help="Random shift peak boundary",
    )
    parser.add_argument(
        "--mask_ratio",
        default=0.75,
        type=float,
        help="ratio of the visual tokens/patches need be masked",
    )
    parser.add_argument(
        "--final_bn", default=False, type=bool, help="Whether to BN motifxregion matrix before region embedding"
    )


    parser.add_argument(
        "--input_dim", default=111, type=int, help="input dimension size for backbone"
    )

    parser.add_argument(
        "--output_dim", default=111, type=int, help="output dimension size for backbone"
    )

    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )

    parser.add_argument(
        "--normalize_target",
        action="store_true",
        default=False,
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
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1.5e-4,
        metavar="LR",
        help="learning rate (default: 1.5e-4)",
    )
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
        help="epochs to warmup LR, if scheduler supports",
    )

    # Dataset parameters
    parser.add_argument(
        "--data_path",
        default="/pmglocal/xf2217/get_data/",
        type=str,
        help="dataset path",
    )
    parser.add_argument("--hic_data_path", type=str, help="hic dataset path")
    parser.add_argument("--data_type", default="fetal", type=str, help="dataset type")
    parser.add_argument("--use_natac", action="store_true", default=False)
    parser.add_argument("--data_set", default="Pretrain", type=str)
    parser.add_argument(
        "--eval_data_set",
        default="Pretrain.GBM_eval",
        type=str,
        help="Evaluation dataset path",
    )
    parser.add_argument("--eval_nonzero", action="store_true", default=False)
    parser.add_argument("--eval_tss", action="store_true", default=False)
    parser.add_argument("--leave_out_celltypes", default=None)
    parser.add_argument("--leave_out_chromosomes", default=None)
    parser.add_argument("--use_seq", default=False, action="store_true")
    parser.add_argument("--sampling_step", default=100, type=int)
    parser.add_argument(
        "--spike_in",
        default=0.01,
        type=float,
        help="ratio of spike unsupervised pretraining",
    )

    # engine
    parser.add_argument(
        "--output_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument("--log_dir", default=None, help="path where to tensorboard log")
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--no_auto_resume", action="store_false", dest="auto_resume")
    parser.set_defaults(auto_resume=True)

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem", help="")
    parser.set_defaults(pin_mem=False)
    parser.add_argument("--flash_attn", action="store_true", default=False, help="flash attention")

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

    return parser


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        num_motif=args.num_motif,
        motif_dim=args.input_dim,
        num_region_per_sample=args.num_region_per_sample,
        encoder_in_chans=args.input_dim,
        encoder_num_classes=args.output_dim,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        output_dim=args.output_dim,
        flash_attn=args.flash_attn,
        final_bn=args.final_bn,
    )

    return model


def main(args):
    utils.init_distributed_mode(args)


    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    model = get_model(args)
    num_region_per_sample = args.num_region_per_sample
    num_motif = args.num_motif
    motif_dim=args.input_dim
    print("Region size = %s" % str(num_region_per_sample))
    print("Number motif = %s" % str(num_motif))
    print("Dimension motif = %s" % str(motif_dim))
    args.region_size = num_region_per_sample

    # get dataset
    sequence_obj = DenseZarrIO(f'{args.data_path}/hg38.zarr', dtype='int8')
    sequence_obj.load_to_memory_dense()

    dataset_train = build_dataset(is_train=True, args=args, sequence_obj=sequence_obj)
    dataset_eval = build_dataset(is_train=False, args=args, sequence_obj=sequence_obj)
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_rank = global_rank
        num_training_steps_per_epoch = (
            len(dataset_train) // args.batch_size // num_tasks
        )

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
        )
        sampler_eval = torch.utils.data.DistributedSampler(
            dataset_eval, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_eval = torch.utils.data.RandomSampler(dataset_eval)
    
    log_writer = None
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    elif global_rank == 0 and args.wandb_project_name is not None:
        wandb.login()
        run = wandb.init(
            project=args.wandb_project_name,
            name=args.wandb_run_name,
        )
        log_writer = utils.WandBLogger(run)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn = get_rev_collate_fn,
        worker_init_fn=worker_init_fn_get,
    )

    data_loader_eval = torch.utils.data.DataLoader(
        dataset_eval,
        sampler=sampler_eval,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn = get_rev_collate_fn,
        worker_init_fn=worker_init_fn_get,
    )

    model.to(device)

    # compile model using torch.compile
    if args.compile_model:
        model = torch.compile(model)
    # ddp 
    

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print("number of params: {} M".format(n_parameters / 1e6))

    total_batch_size = args.batch_size * utils.get_world_size()
    args.lr = args.lr * total_batch_size / 256

    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print(
        "Number of training examples per epoch = %d"
        % (total_batch_size * num_training_steps_per_epoch)
    )

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
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
    )

    if args.normalize_target:
        print("Normalize target!")

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
            np.random.seed(epoch + args.seed)
            torch.manual_seed(epoch + args.seed)
        if log_writer is not None and isinstance(log_writer, utils.TensorboardLogger):
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            normalize_target=args.normalize_target,
        )
        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                )
        if (
            data_loader_eval is not None and epoch % 5 == 0
        ):
            print("*******Start evaluation********")
            test_stats = evaluate_pretrain(
                data_loader_eval, model, device, args, epoch, printlog=True
            )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if utils.is_main_process():
            wandb.log(log_stats)
        
                
        if args.output_dir and utils.is_main_process():
            if log_writer is not None and isinstance(log_writer, utils.TensorboardLogger):
                log_writer.flush()
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

            

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    opts = get_args_parser()
    opts = opts.parse_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
