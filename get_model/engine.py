# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import logging
import math
import sys
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import get_model.utils as utils
from metrics import score_pearsonr, score_r2, score_spearmanr
from timm.data import Mixup
from timm.utils import ModelEma
from get_model.dataset.zarr_dataset import get_mask_pos, get_padding_pos


def pretrain_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    normalize_target: bool = True,
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

    for step, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        sample_track, peak_seq, sample_metadata, celltype_peaks, sample_track_boundary, sample_peak_sequence_boundary, chunk_size, mask, n_peaks, max_n_peaks, total_peak_len, motif_mean_std, _, _ = batch
        if min(chunk_size)<0:
            continue
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]


        sample_track = sample_track.to(device, non_blocking=True).bfloat16()
        peak_seq = peak_seq.to(device, non_blocking=True).bfloat16()
        # mask has 0, 1, -10000
        mask_for_loss = get_mask_pos(mask)
        padding_mask = get_padding_pos(mask)
        mask_for_loss = mask_for_loss.to(device, non_blocking=True).bool()
        padding_mask = padding_mask.to(device, non_blocking=True).bool()
        motif_mean_std = motif_mean_std.to(device, non_blocking=True).bfloat16()
        # chunk_size = chunk_size.to(device, non_blocking=True)
        n_peaks = n_peaks.to(device, non_blocking=True)


        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            output_masked, _, target = model(peak_seq, sample_track, mask_for_loss, padding_mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std)

            # target generation
            with torch.no_grad():
                unnorm_targets = target
                if normalize_target:
                    print('normalize target')
                    regions_squeeze = unnorm_targets
                    regions_norm = (
                        regions_squeeze - regions_squeeze.mean(dim=-2, keepdim=True)
                    ) / (
                        regions_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt()
                        + 1e-6
                    )
                    # we find that the mean is about 0.48 and standard deviation is about 0.08.
                    regions_embed = regions_norm
                else:
                    regions_embed = unnorm_targets

                B, _, C = regions_embed.shape

            mask_for_loss = mask_for_loss.unsqueeze(-1)
            loss_masked_value = loss_masked(input=output_masked*mask_for_loss, target=regions_embed*mask_for_loss)
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

        if log_writer is not None and isinstance(log_writer, utils.TensorboardLogger):
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()
        elif log_writer is not None and isinstance(log_writer, utils.WandBLogger):
            log_writer.update(data={'loss': loss_value,
                 'loss_scale': loss_scale_value,
                 'lr': max_lr,
                 'min_lr': min_lr,
                 'epoch': epoch,
                 'weight_decay': weight_decay_value,
                 'grad_norm': grad_norm},
                step=it, print_freq=print_freq)

        torch.distributed.barrier()
        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    # torch.distributed.barrier()
    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_class_batch(model, peak_seq, sample_track, mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, atac_target, exp_target, other_labels, criterion):
    device = peak_seq.device
    padding_mask = get_padding_pos(mask)
    mask_for_loss = 1-padding_mask
    padding_mask = padding_mask.to(device, non_blocking=True).bool()
    mask_for_loss = mask_for_loss.to(device, non_blocking=True).unsqueeze(-1)
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        atac, exp = model(peak_seq, sample_track, mask_for_loss, padding_mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, other_labels)

    exp = exp * mask_for_loss
    indices = torch.where(mask_for_loss==1)
    exp = exp[indices[0], indices[1], :].flatten()
    exp_target = exp_target * mask_for_loss
    exp_target = exp_target[indices[0], indices[1], :].flatten()
    loss_exp = criterion(exp, exp_target)
    
    if atac is not None:
        atac = atac * mask_for_loss
        indices = torch.where(mask_for_loss==1)
        atac = atac[indices[0], indices[1], :].flatten()
        atac_target = atac_target.unsqueeze(-1) * mask_for_loss
        atac_target = atac_target[indices[0], indices[1], :].flatten()
        loss_atac = criterion(atac, atac_target)
        loss = loss_exp + loss_atac #+ loss_exp
    else:
        loss = loss_exp
    return loss, atac, exp, exp_target 


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

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):  
        # logging.info("data_iter_step: {}".format(data_iter_step))
        # logging.info("start getting batch")
        sample_track, peak_seq, sample_metadata, celltype_peaks, sample_track_boundary, sample_peak_sequence_boundary, chunk_size, mask, n_peaks, max_n_peaks, total_peak_len, motif_mean_std, labels_data, other_labels = batch
        if min(chunk_size)<0:
            continue
        # logging.info("Got batch")
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

        # logging.info("start cuda computing")
        sample_track = sample_track.to(device, non_blocking=True).bfloat16()
        peak_seq = peak_seq.to(device, non_blocking=True).bfloat16()
        motif_mean_std = motif_mean_std.to(device, non_blocking=True).bfloat16()
        # chunk_size = chunk_size.to(device, non_blocking=True)
        n_peaks = n_peaks.to(device, non_blocking=True)
        labels_data = labels_data.to(device, non_blocking=True).bfloat16()
        other_labels = other_labels.to(device, non_blocking=True).bfloat16()
        print(other_labels.shape)
        if loss_scaler is None:
            peaks = peaks.half()
            loss, _, _, _ = train_class_batch(model, peak_seq, sample_track, mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, other_labels[:,:,0], labels_data, other_labels, criterion)
        else:
            loss, _, _, _ = train_class_batch(model, peak_seq, sample_track, mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, other_labels[:,:,0], labels_data, other_labels, criterion)

        loss_value = loss.item()

        # logging.info("Got loss")
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
        # EMA
        if (data_iter_step + 1) % update_freq == 0 and args.eval_each_step:
            test_stats = evaluate_all(
                data_loader_val, model, device, criterion, args, printlog=False
            )
        else:
            test_stats = None

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

        if log_writer is not None and isinstance(log_writer, utils.TensorboardLogger):
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
        elif log_writer is not None and isinstance(log_writer, utils.WandBLogger):
            log_writer.update(data={
                'training_loss': loss_value,
                'loss_scale': loss_scale_value,
                'lr': max_lr,
                'min_lr': min_lr,
                'epoch': epoch,
                'weight_decay': weight_decay_value,
                'grad_norm': grad_norm},
                step=it, print_freq=print_freq)
            if test_stats is not None:
                log_writer.update(data={
                    'r2score': test_stats["r2score"],
                    'pearsonr_score': test_stats["pearsonr_score"],
                    'spearmanr_score': test_stats["spearmanr_score"],
                    'epoch': epoch},
                    step=it, print_freq=print_freq)
        
        # print("Finished one iteration")

    # gather the stats from all processes
    #metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def cal_score_stats(preds, obs, data_loader, args):

    if args.eval_nonzero:
        r2score = score_r2(preds[obs > 0], obs[obs > 0])
        spearmanr_score = score_spearmanr(preds[obs > 0], obs[obs > 0])
        pearsonr_score = score_pearsonr(preds[obs > 0], obs[obs > 0])
    elif args.eval_tss:
        r2score = score_r2(preds, obs)
        spearmanr_score = score_spearmanr(preds, obs)
        pearsonr_score = score_pearsonr(preds, obs)
    else:
        r2score = score_r2(preds, obs)
        spearmanr_score = score_spearmanr(preds, obs)
        pearsonr_score = score_pearsonr(preds, obs)

    return r2score, spearmanr_score, pearsonr_score


@torch.no_grad()
def evaluate_pretrain(data_loader, model, device, args, epoch=0, printlog=True):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()
    output_masked_list = []
    target_list = []
    for i, batch in enumerate(data_loader):
        sample_track, peak_seq, sample_metadata, celltype_peaks, sample_track_boundary, sample_peak_sequence_boundary, chunk_size, mask, n_peaks, max_n_peaks, total_peak_len, motif_mean_std, _, _ = batch
        if min(chunk_size)<0:
            continue
    # for i in tqdm(range(100)):
        mask_for_loss = get_mask_pos(mask)
        padding_mask = get_padding_pos(mask)
        mask_for_loss = mask_for_loss.to(device, non_blocking=True).bool()
        padding_mask = padding_mask.to(device, non_blocking=True).bool()
        peak_seq = peak_seq.bfloat16().cuda()
        sample_track = sample_track.bfloat16().cuda()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            output_masked, _, target = model.forward(peak_seq, sample_track, mask_for_loss, padding_mask, chunk_size, n_peaks.cuda(), max_n_peaks, motif_mean_std.cuda())
        normalize_target = False
        unnorm_targets = target
        if normalize_target:
            regions_squeeze = unnorm_targets
            regions_norm = (
                regions_squeeze - regions_squeeze.mean(dim=-2, keepdim=True)
            ) / (
                regions_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt()
                + 1e-6
            )
            # we find that the mean is about 0.48 and standard deviation is about 0.08.
            regions_embed = regions_norm
        else:
            regions_embed = unnorm_targets

        B, _, C = regions_embed.shape
        mask_for_loss = mask_for_loss.unsqueeze(-1)
        loss_masked = nn.MSELoss()
        loss_masked_value = loss_masked(input=output_masked*mask_for_loss, target=regions_embed*mask_for_loss)
        #loss_atac_value = loss_atac(atac, labels_atac)
        # print(loss_masked_value, loss_atac_value) # masked loss is around 5 times larger than atac loss
        loss = loss_masked_value #+ loss_atac_value * 5
        target = (regions_embed*mask_for_loss).float().detach().cpu().numpy().flatten()
        output_masked = (output_masked*mask_for_loss).float().detach().cpu().numpy().flatten()
        output_masked = output_masked[target>0]
        target = target[target>0]
        output_masked_list.append(output_masked)
        target_list.append(target)
        loss_value = loss.item()
        metric_logger.update(loss=loss_value)

    output_masked_list = np.concatenate(output_masked_list).flatten()
    target_list = np.concatenate(target_list).flatten()
    r2score, pearsonr_score, spearmanr_score = cal_score_stats(output_masked_list, target_list, data_loader, args)

    metric_logger.meters["r2score"].update(r2score, n=1)
    metric_logger.meters["pearsonr_score"].update(pearsonr_score, n=1)
    metric_logger.meters["spearmanr_score"].update(spearmanr_score, n=1)

    if printlog:
        print(
            "* Score@R2 {r2:.3f} Score@pearsonr {pearson:.3f} Score@spearmanr {spearman:.3f}  loss {losses.global_avg:.3f}".format(
                r2=r2score,
                pearson=pearsonr_score,
                spearman=spearmanr_score,
                losses=metric_logger.loss,
            )
        )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    
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
    from tqdm import tqdm
    for batch in tqdm(data_loader):
        sample_track, peak_seq, sample_metadata, celltype_peaks, sample_track_boundary, sample_peak_sequence_boundary, chunk_size, mask, n_peaks, max_n_peaks, total_peak_len, motif_mean_std, labels_data, other_labels = batch
        if min(chunk_size)<0:
            continue
        sample_track = sample_track.to(device, non_blocking=True).bfloat16()
        peak_seq = peak_seq.to(device, non_blocking=True).bfloat16()
        motif_mean_std = motif_mean_std.to(device, non_blocking=True).bfloat16()
        # chunk_size = chunk_size.to(device, non_blocking=True)
        n_peaks = n_peaks.to(device, non_blocking=True)
        labels_data = labels_data.to(device, non_blocking=True).bfloat16()
        other_labels = other_labels.to(device, non_blocking=True).bfloat16()
        # compute output
        loss, atac, exp, exp_target = train_class_batch(model, peak_seq, sample_track, mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std
        , other_labels[:,:,0], labels_data, other_labels, criterion)

        if args.eval_tss:
            padding_mask = get_padding_pos(mask)
            mask_for_loss = 1-padding_mask
            padding_mask = padding_mask.to(device, non_blocking=True).bool()
            mask_for_loss = mask_for_loss.to(device, non_blocking=True).unsqueeze(-1)
            indices = torch.where(mask_for_loss==1)
            # other_labels is B, R, N where [:,:, 1] is TSS indicator
            other_labels_reshape = other_labels[indices[0], indices[1], 1].flatten()
            preds.append(exp.reshape(-1, 2)[other_labels_reshape==1, :].reshape(-1).detach().cpu().numpy())
            obs.append(exp_target.reshape(-1,2)[other_labels_reshape==1, :].reshape(-1).detach().cpu().numpy())
        else:
            preds.append(exp.reshape(-1).detach().cpu().numpy())
            obs.append(exp_target.reshape(-1).detach().cpu().numpy())
            # preds_atac.append(atac.reshape(-1).detach().cpu().numpy())
            # obs_atac.append(atac_targets.reshape(-1).detach().cpu().numpy())

        metric_logger.update(loss=loss.item())

    preds = np.concatenate(preds, axis=0).reshape(-1)
    obs = np.concatenate(obs, axis=0).reshape(-1)
    # preds_atac = np.concatenate(preds_atac, axis=0).reshape(-1)
    # obs_atac = np.concatenate(obs_atac, axis=0).reshape(-1)

    r2score, pearsonr_score, spearmanr_score = cal_score_stats(preds, obs, data_loader, args)
    # r2score_atac, pearsonr_score_atac, spearmanr_score_atac = cal_score_stats(preds_atac, obs_atac, data_loader, args)

    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()

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
