# --------------------------------------------------------
# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import datetime
import io
import json
import math
import os
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import OrderedDict

import hydra
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.distributed as dist
import torch.utils.data
import zarr
from hydra.core.global_hydra import GlobalHydra
from timm.utils import get_state_dict

from get_model.config.config import Config
np.bool = np.bool_
try:
    from torch import inf
except ImportError:
    from torch._six import inf

from tensorboardX import SummaryWriter


def load_config(config_name):
    # Initialize Hydra to load the configuration
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="config", version_base="1.3")
    cfg = hydra.compose(config_name=config_name)
    return cfg

def print_shape(x):
    """a recursive function to print the shape of values in a nested dictionary"""
    if isinstance(x, dict):
        for k, v in x.items():
            print(k)
            print_shape(v)
    else:
        print(x.shape)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total],
                         dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(
                type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


class TensorboardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head="scalar", step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(
                head + "/" + k, v, self.step if step is None else step
            )

    def flush(self):
        self.writer.flush()


class WandBLogger(object):
    def __init__(self, wandb_obj):
        self.wandb = wandb_obj

    def update(self, head="scalar", print_freq=10, step=None, **kwargs):
        if step % print_freq == 0:
            self.wandb.log(kwargs, step=step, commit=True)
        else:
            self.wandb.log(kwargs, step=step, commit=False)

    def flush(self):
        pass


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        args.world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        args.gpu = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        args.dist_url = "tcp://%s:%s" % (
            os.environ["MASTER_ADDR"],
            os.environ["MASTER_PORT"],
        )
        os.environ["LOCAL_RANK"] = str(args.gpu)
        os.environ["RANK"] = str(args.rank)
        os.environ["WORLD_SIZE"] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}): {}, gpu {}".format(
            args.rank, args.dist_url, args.gpu
        ),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def load_checkpoint(checkpoint_path, model_key=None):
    if checkpoint_path.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(checkpoint_path, map_location="cpu", check_hash=True)
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    print("Load ckpt from %s" % checkpoint_path)
    
    if model_key is not None:
        checkpoint_model = None
        for key in model_key.split("|"):
            if key in checkpoint:
                checkpoint_model = checkpoint[key]
                print("Load state_dict by model_key = %s" % key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
    else:
        checkpoint_model = checkpoint
    
    return checkpoint_model

def remove_keys(checkpoint_model, model_state_dict):
    for k in ["head.weight", "head.bias"]:
        if k in checkpoint_model and checkpoint_model[k].shape != model_state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]



def rename_keys(state_dict):
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key
        if "blocks." in new_key:
            new_key = new_key.replace("blocks.", "encoder.blocks.")
        if "fc_norm." in new_key:
            new_key = new_key.replace("fc_norm.", "encoder.norm.")
        if "head." in new_key:
            new_key = new_key.replace("head.", "head_exp.head.")
        if "region_embed.proj." in new_key:
            new_key = new_key.replace(
                "region_embed.proj.", "region_embed.embed.")
        new_state_dict[new_key] = state_dict[key]

    if 'region_embed.embed.weight' in new_state_dict:
        # .unsqueeze(2)
        new_state_dict['region_embed.embed.weight'] = new_state_dict['region_embed.embed.weight']
    return new_state_dict


def rename_lit_state_dict(state_dict, patterns_to_drop=[]):
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace("model.", "")
        new_state_dict[new_key] = state_dict[key]
        for key_to_drop in patterns_to_drop:
            if key_to_drop in new_key:
                del new_state_dict[new_key]
    return new_state_dict


def rename_v1_pretrain_keys(state_dict):
    """
    Rename the keys in the state dictionary.
    """
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace("encoder.head.", "head_mask.")
        new_key = new_key.replace(
            "encoder.region_embed", "region_embed")
        new_key = new_key.replace(
            "region_embed.proj.", "region_embed.embed.")
        new_key = new_key.replace(
            "encoder.cls_token", "cls_token")

        new_state_dict[new_key] = state_dict[key]
    return new_state_dict


def rename_v1_finetune_keys(state_dict):
    """
    Rename the keys in the state dictionary.
    """
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace("blocks.", "encoder.blocks.")
        new_key = new_key.replace("fc_norm.", "encoder.norm.")
        new_key = new_key.replace("encoder.head.", "head_mask.")
        new_key = new_key.replace(
            "encoder.region_embed", "region_embed")
        new_key = new_key.replace(
            "region_embed.proj.", "region_embed.embed.")
        new_key = new_key.replace(
            "encoder.cls_token", "cls_token")
        new_key = new_key.replace(
            "head.", "head_exp.head.")

        new_state_dict[new_key] = state_dict[key]
    return new_state_dict


def extract_state_dict(checkpoint_model):
    if 'model' in checkpoint_model:
        checkpoint_model = checkpoint_model['model']
    if 'state_dict' in checkpoint_model:
        checkpoint_model = checkpoint_model['state_dict']
    return checkpoint_model

def rename_state_dict(state_dict, rename_config):
    if rename_config is None:
        return state_dict
    
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        for pattern, replacement in rename_config.items():
            new_key = new_key.replace(pattern, replacement)
        new_state_dict[new_key] = value
    
    return new_state_dict

def freeze_layers(model, freeze_last_layer=False, freeze_atac_attention=False):
    if freeze_last_layer:
        for name, param in model.named_parameters():
            if not (
                name.startswith("blocks.11")
                or name.startswith("head.")
                or name.startswith("fc_norm")
                or name.startswith("norm")
            ):
                print(name)
                param.requires_grad = False

    if freeze_atac_attention:
        for name, param in model.named_parameters():
            if "atac_attention" in name:
                param.requires_grad = False
                print(f"Freezed weights of {name}")

def load_state_dict(model, state_dict, strict=True, patterns_to_drop=[]):
    # Remove keys matching the patterns_to_drop
    for pattern in patterns_to_drop:
        state_dict = {k: v for k, v in state_dict.items() if pattern not in k}
    model.load_state_dict(state_dict, strict=strict)



def recursive_detach(tensors):
    if isinstance(tensors, dict):
        return {k: recursive_detach(v) for k, v in tensors.items()}
    elif isinstance(tensors, list):
        return [recursive_detach(v) for v in tensors]
    elif isinstance(tensors, torch.Tensor):
        if tensors.is_cuda:
            return tensors.detach().cpu()
    else:
        return tensors

def recursive_numpy(tensors):
    if isinstance(tensors, dict):
        return {k: recursive_numpy(v) for k, v in tensors.items()}
    elif isinstance(tensors, list):
        return [recursive_numpy(v) for v in tensors]
    elif isinstance(tensors, torch.Tensor):
        return tensors.detach().cpu().float().numpy()
    else:
        return tensors

def recursive_save_to_zarr(zarr_group, dict_data, **kwargs):
    for k, v in dict_data.items():
        if isinstance(v, dict):
            subgroup = zarr_group.require_group(k)
            recursive_save_to_zarr(subgroup, v, **kwargs)
        else:
            # if group not exist, create it
            if k not in zarr_group:
                zarr_group.create_dataset(k, data=v, **kwargs)
            else: # append to existing group
                # pad to the same shape
                if isinstance(v, np.ndarray) and isinstance(zarr_group[k], zarr.core.Array) and zarr_group[k].shape[1:] != v.shape[1:]:
                    new_shape = [v.shape[0]] + list(zarr_group[k].shape[1:])
                    new_data = np.zeros(new_shape, dtype=v.dtype)
                    new_data[:, :v.shape[1]] = v
                    zarr_group[k].append(new_data)
                else:
                    zarr_group[k].append(v)
                


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        parameters=None,
        create_graph=False,
        update_grad=True,
    ):
        print(loss)
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device)
                         for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device)
                 for p in parameters]
            ),
            norm_type,
        )
    return total_norm


def cosine_scheduler(
    base_value,
    final_value,
    epochs,
    niter_per_ep,
    warmup_epochs=0,
    start_warmup_value=0,
    warmup_steps=-1,
):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(
            start_warmup_value, base_value, warmup_iters)
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [
            final_value
            + 0.5
            * (base_value - final_value)
            * (1 + math.cos(math.pi * i / (len(iters))))
            for i in iters
        ]
    )
    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def save_model(
    args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None
):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ("checkpoint-%s.pth" % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "scaler": loss_scaler.state_dict(),
                "args": args,
            }

            if model_ema is not None:
                to_save["model_ema"] = get_state_dict(model_ema)

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {"epoch": epoch}
        if model_ema is not None:
            client_state["model_ema"] = get_state_dict(model_ema)
        model.save_checkpoint(
            save_dir=args.output_dir,
            tag="checkpoint-%s" % epoch_name,
            client_state=client_state,
        )



def auto_load_model(
    args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None
):
    output_dir = Path(args.output_dir)
    if loss_scaler is not None:
        # torch.amp
        if args.auto_resume and len(args.resume) == 0:
            import glob

            all_checkpoints = glob.glob(
                os.path.join(output_dir, "checkpoint-*.pth"))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split("-")[-1].split(".")[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(
                    output_dir, "checkpoint-%d.pth" % latest_ckpt
                )
            print("Auto resume checkpoint: %s" % args.resume)

        if args.resume:
            if args.resume.startswith("https"):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location="cpu", check_hash=True
                )
            else:
                checkpoint = torch.load(args.resume, map_location="cpu")
            model_without_ddp.load_state_dict(
                rename_keys(checkpoint["model"]), strict=False)

            print("Resume checkpoint %s" % args.resume)
            if "optimizer" in checkpoint and "epoch" in checkpoint:
                for group in checkpoint['optimizer']['param_groups']:
                    group.setdefault('differentiable', False)
                    group.setdefault('fused', None)
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    args.start_epoch = checkpoint["epoch"] + 1
                except:
                    print("Can't load optimizer state dict!")

                if hasattr(args, "model_ema") and args.model_ema:
                    _load_checkpoint_for_ema(
                        model_ema, checkpoint["model_ema"])
                if "scaler" in checkpoint:
                    loss_scaler.load_state_dict(checkpoint["scaler"])
                print("With optim & sched!")
    else:
        # deepspeed, only support '--auto_resume'.
        if args.auto_resume:
            import glob

            all_checkpoints = glob.glob(
                os.path.join(output_dir, "checkpoint-*"))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split("-")[-1].split(".")[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(
                    output_dir, "checkpoint-%d" % latest_ckpt)
                print("Auto resume checkpoint: %d" % latest_ckpt)
                _, client_states = model.load_checkpoint(
                    args.output_dir, tag="checkpoint-%d" % latest_ckpt
                )
                args.start_epoch = client_states["epoch"] + 1
                if model_ema is not None:
                    if args.model_ema:
                        _load_checkpoint_for_ema(
                            model_ema, client_states["model_ema"])


def create_ds_config(args):
    args.deepspeed_config = os.path.join(
        args.output_dir, "deepspeed_config.json")
    with open(args.deepspeed_config, mode="w") as writer:
        ds_config = {
            "train_batch_size": args.batch_size * args.update_freq * get_world_size(),
            "train_micro_batch_size_per_gpu": args.batch_size,
            "steps_per_print": 1000,
            "optimizer": {
                "type": "Adam",
                "adam_w_mode": True,
                "params": {
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "bias_correction": True,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                },
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 7,
                "loss_scale_window": 128,
            },
        }

        writer.write(json.dumps(ds_config, indent=2))
