# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import sys
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import utils
from metrics import score_pearsonr, score_r2, score_spearmanr
from timm.data import Mixup
from timm.utils import ModelEma


def pretrain_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    normlize_target: bool = True,
    log_writer=None,
    lr_scheduler=None,
    start_steps=0,
    lr_schedule_values=None,
    wd_schedule_values=None,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "min_lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    loss_func = nn.MSELoss()

    for step, (batch, _) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        regions, bool_masked_pos, ctcf_pos = batch
        regions = regions.to(device, non_blocking=True)
        regions = regions.float()
        bool_masked_pos = (
            bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
        )

        with torch.no_grad():
            unnorm_regions = regions

            if normlize_target:
                regions_squeeze = unnorm_regions
                regions_norm = (
                    regions_squeeze - regions_squeeze.mean(dim=-2, keepdim=True)
                ) / (
                    regions_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt()
                    + 1e-6
                )
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                regions_embed = regions_norm
            else:
                regions_embed = unnorm_regions

            B, _, C = regions_embed.shape
            labels = regions_embed[bool_masked_pos].reshape(B, -1, C)

        outputs = model(regions, bool_masked_pos)
        loss = loss_func(input=outputs, target=labels)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = (
            hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        )
        grad_norm = loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=is_second_order,
        )
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_class_batch(model, samples, target, criterion):
    outputs = model(samples)
    loss = criterion(outputs, target)
    return loss, outputs


def finetune_train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    data_loader_val: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    start_steps=None,
    lr_schedule_values=None,
    wd_schedule_values=None,
    num_training_steps_per_epoch=None,
    update_freq=None,
    args=None,
):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "min_lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets, cells) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if (
            lr_schedule_values is not None
            or wd_schedule_values is not None
            and data_iter_step % update_freq == 0
        ):
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        samples = samples.float()
        targets = targets.to(device, non_blocking=True)
        targets = targets.float()

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(model, samples, targets, criterion)
        else:
            loss, output = train_class_batch(model, samples, targets, criterion)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = (
                hasattr(optimizer, "is_second_order") and optimizer.is_second_order
            )
            loss /= update_freq
            grad_norm = loss_scaler(
                loss,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=is_second_order,
                update_grad=(data_iter_step + 1) % update_freq == 0,
            )
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        # NOTE: evaluation

        if (data_iter_step + 1) % update_freq == 0 and args.eval_each_step:
            test_stats = evaluate_all(
                data_loader_val, model, device, args, printlog=False
            )
        else:
            test_stats = None
        # if mixup_fn is None:
        #     r2score = score_r2(output, targets)
        #     spearmanr_score = score_spearmanr(output, targets)
        # else:
        #     r2score = None
        #     spearmanr_score = None

        metric_logger.update(loss=loss_value)
        if test_stats is not None:
            metric_logger.update(r2score=test_stats["r2score"])
            metric_logger.update(pearsonr_score=test_stats["pearsonr_score"])
            metric_logger.update(spearmanr_score=test_stats["spearmanr_score"])
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            if test_stats is not None:
                log_writer.update(r2score=test_stats["r2score"], head="loss")
                log_writer.update(
                    pearsonr_score=test_stats["pearsonr_score"], head="loss"
                )
                log_writer.update(
                    spearmanr_score=test_stats["spearmanr_score"], head="loss"
                )

            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_all(data_loader, model, device, args, epoch=0, printlog=True):
    criterion = torch.nn.MSELoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()

    preds = []
    obs = []

    for batch in data_loader:
        samples = batch[0].float()
        target = batch[1].float()
        samples = samples.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.amp.autocast("cuda" if args.cuda else "cpu"):
            output = model(samples)
            loss = criterion(output, target)
            # print("loss:", loss)

        preds.append(output.reshape(-1).detach().cpu().numpy())
        obs.append(target.reshape(-1).detach().cpu().numpy())

        metric_logger.update(loss=loss.item())

    preds = np.concatenate(preds, axis=0).reshape(-1)
    obs = np.concatenate(obs, axis=0).reshape(-1)

    print("preds:", preds.shape)
    print("obs:", obs.shape)

    if args.eval_nonzero:
        r2score = score_r2(preds[obs > 0], obs[obs > 0])
        spearmanr_score = score_spearmanr(preds[obs > 0], obs[obs > 0])
        pearsonr_score = score_pearsonr(preds[obs > 0], obs[obs > 0])
    if args.eval_tss:
        # gene_idx = np.load(os.path.join(args.data_path, "tss_idx/gene_idx_{}.npy".format(args.num_region_per_sample)))
        gene_idx = data_loader.dataset.geneidx.reshape(-1)
        r2score = score_r2(preds[gene_idx], obs[gene_idx])
        spearmanr_score = score_spearmanr(preds[gene_idx], obs[gene_idx])
        pearsonr_score = score_pearsonr(preds[gene_idx], obs[gene_idx])
    else:
        if args.setting != "human":
            r2score = score_r2(preds, obs)
            spearmanr_score = score_spearmanr(preds, obs)
            pearsonr_score = score_pearsonr(preds, obs)
        else:
            r2score = 0
            spearmanr_score = 0
            pearsonr_score = 0
            for i in range(len(preds)):
                r2score += score_r2(preds[i], obs[i])
                pearsonr_score += score_pearsonr(preds[i], obs[i])
                spearmanr_score += score_spearmanr(preds[i], obs[i])

            r2score = r2score / len(preds)
            spearmanr_score = spearmanr_score / len(preds)
            pearsonr_score = pearsonr_score / len(preds)

            # r2score = np.mean([score_r2(preds[i], obs[i]) for i in range(len(preds))])
            # spearmanr_score = np.mean([score_spearmanr(preds[i], obs[i]) for i in range(len(preds))])
            # pearsonr_score = np.mean([score_pearsonr(preds[i], obs[i]) for i in range(len(preds))])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    metric_logger.meters["r2score"].update(r2score, n=1)
    metric_logger.meters["pearsonr_score"].update(pearsonr_score, n=1)
    metric_logger.meters["spearmanr_score"].update(spearmanr_score, n=1)

    if printlog:
        print(
            "* Score@R2 {r2:.3f} Score@pearsonr {pearson:.3f} Score@spearmanr {spearman:.3f} loss {losses.global_avg:.3f}".format(
                r2=r2score,
                pearson=pearsonr_score,
                spearman=spearmanr_score,
                losses=metric_logger.loss,
            )
        )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
