import argparse
import datetime
import json
import os
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import utils
from dataset.dataset import build_dataset#, sparse_batch_collate
from engine import evaluate_all, finetune_train_one_epoch as train_one_epoch
from optim import (
    LayerDecayValueAssigner,
    create_optimizer,
    get_parameter_groups,
)
from timm.data.mixup import Mixup
from timm.models import create_model
from timm.utils import ModelEma
from utils import NativeScalerWithGradNormCount as NativeScaler

import get_model.model.model
import wandb
import warnings

# Suppress FutureWarnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
torch.autograd.set_detect_anomaly(True)


def get_args():
    parser = argparse.ArgumentParser(
        "GeneFormer fine-tuning and evaluation script for gene expression prediction",
        add_help=False,
    )
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--update_freq", default=1, type=int)
    parser.add_argument("--save_ckpt_freq", default=20, type=int)

    parser.add_argument("--setting", default="ood", type=str)
    parser.add_argument(
        "--spike_in", default=1.0, type=float, help="ratio of spike fetal finetuning"
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="deit_base_patch16_224",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )

    parser.add_argument("--input_size", default=224, type=int, help="images input size")
    parser.add_argument("--input_dim", default=111, type=int, help="input dim")
    parser.add_argument(
        "--last_layer", default=False, type=bool, help="train only last layers"
    )
    parser.add_argument(
        "--drop",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Dropout rate (default: 0.)",
    )
    parser.add_argument(
        "--attn_drop_rate",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Attention dropout rate (default: 0.)",
    )
    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )

    parser.add_argument(
        "--disable_eval_during_finetuning", action="store_true", default=False
    )

    parser.add_argument("--model_ema", action="store_true", default=False)
    parser.add_argument("--model_ema_decay", type=float, default=0.9999, help="")
    parser.add_argument(
        "--model_ema_force_cpu", action="store_true", default=False, help=""
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
        default=1e-6,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0 (1e-5)",
    )

    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
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

    # Augmentation parameters
    parser.add_argument(
        "--color_jitter",
        type=float,
        default=0.4,
        metavar="PCT",
        help="Color jitter factor (default: 0.4)",
    )

    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )
    parser.add_argument(
        "--train_interpolation",
        type=str,
        default="bicubic",
        help='Training interpolation (random, bilinear, bicubic default: "bicubic")',
    )

    # Evaluation parameters
    parser.add_argument("--crop_pct", type=float, default=None)
    parser.add_argument(
        "--criterion",
        default="mse",
        type=str,
        metavar="CRITERION",
        help='Criterion (default: "mse"',
    )

    # * Random Erase params
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        default=False,
        help="Do not random erase first (clean) augmentation split",
    )

    # * Finetuning params
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")
    parser.add_argument("--model_key", default="model|module", type=str)
    parser.add_argument("--model_prefix", default="", type=str)
    parser.add_argument("--init_scale", default=0.001, type=float)
    parser.add_argument("--use_mean_pooling", action="store_true")
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument("--use_cls", action="store_false", dest="use_mean_pooling")

    # Dataset parameters
    parser.add_argument(
        "--data_path",
        default="/datasets01/imagenet_full_size/061417/",
        type=str,
        help="dataset path",
    )
    parser.add_argument(
        "--eval_data_path", default=None, type=str, help="dataset path for evaluation"
    )
    parser.add_argument("--nb_classes", default=1, type=int, help="1 for regression")
    parser.add_argument(
        "--num_region_per_sample",
        default=600,
        type=int,
        help="number of regions for each sample",
    )

    parser.add_argument(
        "--data_set",
        default="Expression",
        choices=["Expression", "Expression_v2", "Expression_v3", "TFBS"],
        type=str,
        help="ImageNet dataset path",
    )
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

    parser.add_argument("--save_ckpt", action="store_true")
    parser.add_argument("--no_save_ckpt", action="store_false", dest="save_ckpt")
    parser.set_defaults(save_ckpt=True)

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--eval_freq",
        default=5,
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
    parser.add_argument("--split_ratio", default=0.7, type=float)
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    parser.add_argument("--enable_deepspeed", action="store_true", default=False)

    parser.add_argument(
        "--vis_attn", action="store_true", help="Perform attention visualization only"
    )
    parser.add_argument(
        "--input_x_attn",
        action="store_true",
        help="Perform input*attention visualization",
    )
    parser.add_argument("--eval_each_step", action="store_true", default=False)
    parser.add_argument("--eval_nonzero", action="store_true", default=False)
    parser.add_argument("--eval_tss", action="store_true", default=False)
    parser.add_argument("--target_type", default="Log", type=str)
    parser.add_argument("--target_thresh", default=32, type=int)
    parser.add_argument(
        "--data_type",
        default="fetal",
        type=str,
        help="dataset type",
    )
    # parser.add_argument('--spike_in', default=1, type=float, help='spike in ratio')
    parser.add_argument("--plot_scatter", action="store_true", default=False)
    parser.add_argument("--use_natac", action="store_true", default=False)
    parser.add_argument("--mask_tss", action="store_true", default=False)
    parser.add_argument("--leave_out_celltypes", default="Astrocytes", type=str)
    parser.add_argument("--leave_out_chromosomes", default="chr4", type=str)
    parser.add_argument("--use_seq", default=False, action="store_true")
    parser.add_argument("--sampling_step", default=100, type=int)
    parser.add_argument("--target_sequence_length", default=200, type=int)

    # wandb params
    parser.add_argument("--wandb_project_name", type=str, default="get-finetune", help="wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name")
    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig

            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed==0.4.0'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init


def main(args, ds_init):
    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)

    print(args)

    if utils.is_main_process(): # Log metrics only on main process
        wandb.login()
        run = wandb.init(
            project=opts.wandb_project_name,
            name=opts.wandb_run_name,
        )

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True
    dataset_train = build_dataset(is_train=True, args=args)
    if args.disable_eval_during_finetuning:
        dataset_val = None
    else:
        dataset_val = build_dataset(is_train=False, args=args)

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

    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        # collate_fn=sparse_batch_collate,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            # collate_fn=sparse_batch_collate,
        )
    else:
        data_loader_val = None

    mixup_fn = None
    # mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    # if mixup_active:
    #     print("Mixup is activated!")
    #     mixup_fn = Mixup(
    #         mixup_alpha=args.mixup,
    #         cutmix_alpha=args.cutmix,
    #         cutmix_minmax=args.cutmix_minmax,
    #         prob=args.mixup_prob,
    #         switch_prob=args.mixup_switch_prob,
    #         mode=args.mixup_mode,
    #         label_smoothing=args.smoothing,
    #         num_classes=args.nb_classes,
    #     )

    model = create_model(
        args.model,
        pretrained=False,
        in_chans=args.input_dim,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        setting=args.setting,
    )

    num_region_per_sample = args.num_region_per_sample
    print("Region size = %s" % str(num_region_per_sample))

    if args.finetune:
        if args.finetune.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.finetune, map_location="cpu")

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split("|"):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias"]:
            if (
                k in checkpoint_model
                and checkpoint_model[k].shape != state_dict[k].shape
            ):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith("backbone."):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith("encoder."):
                new_dict[key[8:]] = checkpoint_model[key]
            else:
                new_dict[key] = checkpoint_model[key]
        checkpoint_model = new_dict

        # interpolate position embedding
        # if "pos_embed" in checkpoint_model:
        #     pos_embed_checkpoint = checkpoint_model["pos_embed"]
        #     embedding_size = pos_embed_checkpoint.shape[-1]
        #     num_patches = model.patch_embed.num_patches
        #     num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        #     # height (== width) for the checkpoint position embedding
        #     orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        #     # height (== width) for the new position embedding
        #     new_size = int(num_patches**0.5)
        #     # class_token and dist_token are kept unchanged
        #     if orig_size != new_size:
        #         print(
        #             "Position interpolate from %dx%d to %dx%d"
        #             % (orig_size, orig_size, new_size, new_size)
        #         )
        #         extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        #         # only the position tokens are interpolated
        #         pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        #         pos_tokens = pos_tokens.reshape(
        #             -1, orig_size, orig_size, embedding_size
        #         ).permute(0, 3, 1, 2)
        #         pos_tokens = torch.nn.functional.interpolate(
        #             pos_tokens,
        #             size=(new_size, new_size),
        #             mode="bicubic",
        #             align_corners=False,
        #         )
        #         pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        #         new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        #         checkpoint_model["pos_embed"] = new_pos_embed

        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
        if args.last_layer:
            for name, param in model.named_parameters():
                if not (
                    name.startswith("blocks.11")
                    or name.startswith("head.")
                    or name.startswith("fc_norm")
                    or name.startswith("norm")
                ):
                    print(name)
                    param.requires_grad = False

        # model.load_state_dict(checkpoint_model, strict=False)

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else "",
            resume="",
        )
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

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

    num_layers = model_without_ddp.num_layers
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            list(
                args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)
            )
        )
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()
    print("Skip weight decay list: ", skip_weight_decay_list)

    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model,
            args.weight_decay,
            skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None,
        )
        model, optimizer, _, _ = ds_init(
            args=args,
            model=model,
            model_parameters=optimizer_params,
            dist_init_required=not args.distributed,
        )

        print(
            "model.gradient_accumulation_steps() = %d"
            % model.gradient_accumulation_steps()
        )
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu],
                find_unused_parameters=True,
            )
            model_without_ddp = model.module
        print("Complete DistributedDataParallel!")

        optimizer = create_optimizer(
            args,
            model_without_ddp,
            skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None,
            get_layer_scale=assigner.get_scale if assigner is not None else None,
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

    if args.criterion == "mse":
        criterion = torch.nn.MSELoss(reduction="sum")
    elif args.criterion == "poisson":
        criterion = torch.nn.PoissonNLLLoss(log_input=False, reduction="mean")

    print("criterion = %s" % str(criterion))

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

    # if args.vis_attn:
    #     test_stats = evaluate_attn(data_loader_val, model, device, args)
    #     max_r2score = test_stats["r2score"]
    #     max_pearsonr_score = test_stats["pearsonr_score"]
    #     max_spearmanr_score = test_stats["spearmanr_score"]

    #     print(f"Statistics of the network on the {len(dataset_val)} test images")
    #     print(f'R2score: {max_r2score:.3f}, pearsonr_score: {max_pearsonr_score:.3f}, spearmanr_score: {max_spearmanr_score:.3f}')
    #     exit(0)

    # if args.input_x_attn:
    #     test_stats = evaluate_input_x_attn(data_loader_val, model, device, args)
    #     max_r2score = test_stats["r2score"]
    #     max_pearsonr_score = test_stats["pearsonr_score"]
    #     max_spearmanr_score = test_stats["spearmanr_score"]

    #     print(f"Statistics of the network on the {len(dataset_val)} test images")
    #     print(f'R2score: {max_r2score:.3f}, pearsonr_score: {max_pearsonr_score:.3f}, spearmanr_score: {max_spearmanr_score:.3f}')
    #     exit(0)

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
            criterion,
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
            # print(
            #     f"R2, Pearson, Spearmanr Score of the network on the {len(dataset_val)} test atac: {test_stats['r2score_atac']:.1f}, {test_stats['pearsonr_score_atac']:.1f}, {test_stats['spearmanr_score_atac']:.1f}"
            # )

            if max_r2score < test_stats["r2score"]:
                max_r2score = test_stats["r2score"]
                max_pearsonr_score = test_stats["pearsonr_score"]
                max_spearmanr_score = test_stats["spearmanr_score"]

                # max_r2score_atac = test_stats["r2score_atac"]
                # max_pearsonr_score_atac = test_stats["pearsonr_score_atac"]
                # max_spearmanr_score_atac = test_stats["spearmanr_score_atac"]

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
            # print(
            #     f"Max r2score atac: {max_r2score_atac:.3f}, Max pearsonr_score atac: {max_pearsonr_score_atac:.3f}, Max spearmanr_score atac: {max_spearmanr_score_atac:.3f}"
            # )
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
                # log_writer.update(
                #     test_r2score_atac=test_stats["r2score_atac"], head="perf", step=epoch
                # )
                # log_writer.update(
                #     test_spearmanr_score_atac=test_stats["pearsonr_score_atac"],
                #     head="perf",
                #     step=epoch,
                # )
                # log_writer.update(
                #     test_spearmanr_score_atac=test_stats["spearmanr_score_atac"],
                #     head="perf",
                #     step=epoch,
                # )
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
                # **{f'test_{k}': v for k, v in test_stats.items()},
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


if __name__ == "__main__":
    opts, ds_init = get_args()

    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)
