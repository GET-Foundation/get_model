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

    loss_masked = nn.MSELoss()
    #loss_atac = nn.PoissonNLLLoss(log_input=False, reduction="mean")

    for step, (batch) in enumerate(
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

        regions, seq, bool_masked_pos, ctcf_pos = batch
        regions = regions.to(device, non_blocking=True)
        seq = seq.to(device, non_blocking=True)
        regions = regions.float()
        bool_masked_pos = (
            bool_masked_pos.to(device, non_blocking=True).bool()
        )
        ctcf_pos = ctcf_pos.to(device, non_blocking=True).bool()

        with torch.no_grad():
            unnorm_regions = regions
            labels_atac = unnorm_regions[:,:,-1]
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
            labels_masked = regions_embed[bool_masked_pos].reshape(B, -1, C)


        output_masked, atac = model(regions, seq, bool_masked_pos, ctcf_pos)
        loss_masked_value = loss_masked(input=output_masked, target=labels_masked)
        #loss_atac_value = loss_atac(atac, labels_atac)
        # print(loss_masked_value, loss_atac_value) # masked loss is around 5 times larger than atac loss
        loss = loss_masked_value #+ loss_atac_value * 5

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
    #metric_logger.synchronize_between_processes()
    #print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_class_batch(model, peak, seq, mask, ctcf_pos, atac_target, exp_target, criterion):
    atac, exp = model(peak, seq, mask, ctcf_pos)
    # loss_atac = criterion(atac, atac_target)
    loss_exp = criterion(exp, exp_target)
    # loss = loss_atac + loss_exp
    loss = loss_exp
    return loss, atac, exp


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

    for data_iter_step, (peaks, seq, mask, ctcf_pos, exp_targets) in enumerate(
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

        peaks = peaks.to(device, non_blocking=True)
        peaks = peaks.float()
        seq = seq.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True).bool().squeeze(-1)
        exp_targets = exp_targets.to(device, non_blocking=True)
        exp_targets = exp_targets.float()
        ctcf_pos = ctcf_pos.to(device, non_blocking=True).bool()
        atac_targets = peaks[:, :, -1]

        if loss_scaler is None:
            peaks = peaks.half()
            loss, atac, exp = train_class_batch(model, peaks, seq, mask, ctcf_pos, atac_targets, exp_targets, criterion)
        else:
            loss, atac, exp = train_class_batch(model, peaks, seq, mask, ctcf_pos, atac_targets, exp_targets, criterion)

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
                data_loader_val, model, device, criterion, args, printlog=False
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
    #metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def cal_score_stats(preds, obs, data_loader, args, eval_nonzero=False, eval_tss=False):

    if eval_nonzero:
        r2score = score_r2(preds[obs > 0], obs[obs > 0])
        spearmanr_score = score_spearmanr(preds[obs > 0], obs[obs > 0])
        pearsonr_score = score_pearsonr(preds[obs > 0], obs[obs > 0])
    if eval_tss:
        gene_idx = np.where(data_loader.dataset.tssidxs.reshape(-1)>0)[0]
        r2score = score_r2(preds[gene_idx][obs[gene_idx]>0], obs[gene_idx][obs[gene_idx]>0])
        spearmanr_score = score_spearmanr(preds[gene_idx][obs[gene_idx]>0], obs[gene_idx][obs[gene_idx]>0])
        pearsonr_score = score_pearsonr(preds[gene_idx][obs[gene_idx]>0], obs[gene_idx][obs[gene_idx]>0])
    else:
        r2score = score_r2(preds, obs)
        spearmanr_score = score_spearmanr(preds, obs)
        pearsonr_score = score_pearsonr(preds, obs)

    return r2score, spearmanr_score, pearsonr_score


@torch.no_grad()
def evaluate_all(data_loader, model, device, criterion, args, epoch=0, printlog=True):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()

    preds = []
    obs = []
    preds_atac = []
    obs_atac = []

    for (peaks, seq, mask, ctcf_pos, exp_targets) in data_loader:
        peaks = peaks.to(device, non_blocking=True)
        peaks = peaks.float()
        seq = seq.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True).bool().squeeze(-1)
        exp_targets = exp_targets.to(device, non_blocking=True)
        exp_targets = exp_targets.float()
        ctcf_pos = ctcf_pos.to(device, non_blocking=True).bool()
        atac_targets = peaks[:, :, -1]
        # compute output
        # with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        atac, exp = model(peaks, seq, mask, ctcf_pos)
        # loss_atac = criterion(atac, atac_targets)
        loss_exp = criterion(exp, exp_targets)
        # loss = loss_atac + loss_exp
        loss = loss_exp

        preds.append(exp.reshape(-1).detach().cpu().numpy())
        obs.append(exp_targets.reshape(-1).detach().cpu().numpy())
        # preds_atac.append(atac.reshape(-1).detach().cpu().numpy())
        # obs_atac.append(atac_targets.reshape(-1).detach().cpu().numpy())

        metric_logger.update(loss=loss.item())

    preds = np.concatenate(preds, axis=0).reshape(-1)
    obs = np.concatenate(obs, axis=0).reshape(-1)
    # preds_atac = np.concatenate(preds_atac, axis=0).reshape(-1)
    # obs_atac = np.concatenate(obs_atac, axis=0).reshape(-1)

    r2score, pearsonr_score, spearmanr_score = cal_score_stats(preds, obs, data_loader, args, eval_tss=True)
    # r2score_atac, pearsonr_score_atac, spearmanr_score_atac = cal_score_stats(preds_atac, obs_atac, data_loader, args)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    metric_logger.meters["r2score"].update(r2score, n=1)
    metric_logger.meters["pearsonr_score"].update(pearsonr_score, n=1)
    metric_logger.meters["spearmanr_score"].update(spearmanr_score, n=1)

    # metric_logger.meters["r2score_atac"].update(r2score_atac, n=1)
    # metric_logger.meters["pearsonr_score_atac"].update(pearsonr_score_atac, n=1)
    # metric_logger.meters["spearmanr_score_atac"].update(spearmanr_score_atac, n=1)

    if printlog:
        print(
            "* Score@R2 {r2:.3f} Score@pearsonr {pearson:.3f} Score@spearmanr {spearman:.3f}  loss {losses.global_avg:.3f}".format(
                r2=r2score,
                pearson=pearsonr_score,
                spearman=spearmanr_score,
                # r2_atac=r2score_atac,
                # pearson_atac=pearsonr_score_atac,
                # spearman_atac=spearmanr_score_atac,
                losses=metric_logger.loss,
            )
        )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
