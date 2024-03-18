# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import logging
import math
import stat
import sys
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import get_model.utils as utils
from get_model.metrics import score_pearsonr, score_r2, score_spearmanr, multinomial_nll
from timm.data import Mixup
from timm.utils import ModelEma
from get_model.dataset.zarr_dataset import get_mask_pos, get_padding_pos


import torch
import torch.nn.functional as F
from torch.distributions import Multinomial


class BaseTrainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, data_loader):
        raise NotImplementedError

    def validate_epoch(self, data_loader):
        raise NotImplementedError

    def train(self, train_data_loader, valid_data_loader, epochs):
        raise NotImplementedError


def calculate_loss_scale_and_get_grad_norm(model, optimizer, loss_scaler, max_norm, loss):
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
    return grad_norm,loss_scale_value

def get_optimizer_weight_decay(optimizer):
    weight_decay_value = None
    for group in optimizer.param_groups:
        if group["weight_decay"] > 0:
            weight_decay_value = group["weight_decay"]
    return weight_decay_value

def get_optimizer_lr_range(optimizer):
    min_lr = 10.0
    max_lr = 0.0
    for group in optimizer.param_groups:
        min_lr = min(min_lr, group["lr"])
        max_lr = max(max_lr, group["lr"])
    return min_lr,max_lr

def apply_schedule_to_optimizer(optimizer, lr_schedule_values, wd_schedule_values, it):
    if lr_schedule_values is not None or wd_schedule_values is not None:
        for i, param_group in enumerate(optimizer.param_groups):
            if lr_schedule_values is not None:
                param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
            if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                param_group["weight_decay"] = wd_schedule_values[it]

def check_loss_value(loss):
    """Check for unusual loss values, e.g. NaN or infinity. If such values are
    encountered, the training is stopped."""
    loss_value = loss.item()
    if not math.isfinite(loss_value):
        print("Loss is {}, stopping training".format(loss_value))
        sys.exit(1)
    return loss_value


def pretrain_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    normalize_target: bool = True,
    loggers=None,
    lr_scheduler=None,
    start_steps=0,
    lr_schedule_values=None,
    wd_schedule_values=None,
):
    model.train()
    
    for step, batch in data_loader:
        batch = model.data_prep(batch)
        # if chunk_size contains negative values, skip this batch
        if min(batch['chunk_size'])<0:
            continue
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        apply_schedule_to_optimizer(optimizer, lr_schedule_values, wd_schedule_values, it)


        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            output = model(**batch)
            pred, obs = model.before_loss(batch, output)
            loss = model.loss_fn(pred, obs)


        loss_value = check_loss_value(loss)
        optimizer.zero_grad()
        grad_norm, loss_scale_value = calculate_loss_scale_and_get_grad_norm(
            model, optimizer, loss_scaler, max_norm, loss)
        torch.cuda.synchronize()
        min_lr, max_lr = get_optimizer_lr_range(optimizer)
        weight_decay_value = get_optimizer_weight_decay(optimizer)

        stats_dict = {
            "loss": loss_value,
            "loss_scale": loss_scale_value,
            "lr": max_lr,
            "min_lr": min_lr,
            "weight_decay": weight_decay_value,
            "grad_norm": grad_norm,
            "epoch": epoch
        }
        if loggers is not None:
            loggers.update(stats_dict, step=it)
        
        torch.distributed.barrier()
        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    return stats_dict


def train_class_batch(model, peak_seq, sample_track, mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, atac_target, exp_target, other_labels, criterion, hic_matrix=None, args=None):
    if 'ChrombpNet' in model._get_name()=='GETFinetuneChrombpNet' or 'chrombpnet' in args.model:
        return train_chrombpnet(model, peak_seq, sample_track, mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, atac_target, exp_target, other_labels, criterion, hic_matrix=hic_matrix)
    else:
        return train_class_batch_exp(model, peak_seq, sample_track, mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, atac_target, exp_target, other_labels, criterion, hic_matrix=hic_matrix)

def train_chrombpnet(model, peak_seq, aprofile_target, loss_mask, padding_mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, atpm_target, exp_target, other_labels, criterion, hic_matrix=None):
    device = peak_seq.device
    loss_mask = 1-padding_mask
    padding_mask = padding_mask.to(device, non_blocking=True).bool()
    loss_mask = loss_mask.to(device, non_blocking=True).unsqueeze(-1)
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        atpm, aprofile = model(peak_seq, aprofile_target, loss_mask, padding_mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, other_labels, hic_matrix=hic_matrix)
    
    B, R, N = atpm.shape
    aprofile, aprofile_target = crop_output(aprofile, aprofile_target, B, R)
    loss_aprofile = multinomial_nll(aprofile_target.float(), aprofile.float())
    
    atpm = atpm * loss_mask
    indices = torch.where(loss_mask==1)
    atpm = atpm[indices[0], indices[1], :].flatten()
    # atpm_target = torch.log10(aprofile_target.sum(2)+1).unsqueeze(-1) * mask_for_loss
    atpm_target = atpm_target.unsqueeze(-1) * loss_mask
    atpm_target = atpm_target[indices[0], indices[1], :].flatten()
    loss_atpm = criterion(atpm.float(), atpm_target.float())
    loss = loss_atpm + loss_aprofile 
    return {'loss': loss, 'loss_atpm': loss_atpm, 'loss_aprofile': loss_aprofile,
            'atpm_pred': atpm, 'atpm_target': atpm_target, 
            'aprofile_pred': aprofile, 'aprofile_target': aprofile_target}

def train_class_batch_exp(model, peak_seq, sample_track, mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, atac_target, exp_target, other_labels, criterion, hic_matrix=None):
    device = peak_seq.device
    padding_mask = get_padding_pos(mask)
    mask_for_loss = 1-padding_mask
    padding_mask = padding_mask.to(device, non_blocking=True).bool()
    mask_for_loss = mask_for_loss.to(device, non_blocking=True).unsqueeze(-1)
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        atac, exp, confidence = model(peak_seq, sample_track, mask_for_loss, padding_mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, other_labels, hic_matrix=hic_matrix)

    B, R, N = exp.shape
    exp = exp * mask_for_loss
    exp_target = exp_target * mask_for_loss
    if confidence is not None:
        cosine_similarity = torch.cosine_similarity(exp.reshape(B, -1), exp_target.reshape(B, -1), dim=-1).detach()
        # use cosine similarity as a target for confidence header, map each element to a bin in (-1,1), 50 bin in total
        # bin 0 is -1, bin 49 is 1
        # bin 0-24 is negative, bin 25 is 0, bin 26-49 is positive
        # get the bin index
        confidence_target = ((cosine_similarity+1)/2*49).long()
        # clip the value to 0-49
        confidence_target = torch.clamp(confidence_target, 0, 49)
        # confidence is (B, R, 1)
        confidence_pred = confidence.mean(1).softmax(dim=-1)
        # cross entropy loss for confidence header
        loss_confidence = nn.CrossEntropyLoss()(confidence_pred, confidence_target)
    else:
        confidence_pred = None
        confidence_target = None
        loss_confidence = None
    indices = torch.where(mask_for_loss==1)
    exp = exp[indices[0], indices[1], :].flatten()
    exp_target = exp_target[indices[0], indices[1], :].flatten()
    loss_exp = criterion(exp, exp_target)

    if atac is not None:
        atac = atac * mask_for_loss
        indices = torch.where(mask_for_loss==1)
        atac = atac[indices[0], indices[1], :].flatten()
        atac_target = atac_target.unsqueeze(-1) * mask_for_loss
        atac_target = atac_target[indices[0], indices[1], :].flatten()
        loss_atac = criterion(atac, atac_target)
        loss = loss_exp + loss_atac 
    else:
        loss = loss_exp
    if confidence is not None:
        loss = loss + loss_confidence * 0.1
    # return loss, exp, exp_target, atac, atac_target, confidence_pred, confidence_target
    return {'loss': loss, 'loss_atac': loss_atac, 'loss_exp': loss_exp, 'loss_confidence': loss_confidence,
            'exp_pred': exp, 'exp_target': exp_target, 
            'atac_pred': atac, 'atac_target': atac_target, 
            'confidence_pred': confidence_pred, 'confidence_target': confidence_target}

def crop_output(aprofile, aprofile_target, B, R, target_length=1000):
    # crop aprofile to center 1000bp, assume the input is (B, R, L)
    B, R, L = aprofile.shape
    if aprofile.shape[1] != aprofile_target.shape[1]:
        current_length = aprofile.shape[2]
        diff_length = current_length - target_length
        assert diff_length % 2 == 0
        crop_size = diff_length // 2
        aprofile = aprofile[:,:,crop_size:crop_size+target_length]
        aprofile_target = aprofile_target.reshape(B, R, -1)
        diff_length = aprofile_target.shape[2] - target_length
        assert diff_length % 2 == 0
        crop_size = diff_length // 2
        aprofile_target = aprofile_target[:,:,crop_size:crop_size+target_length]
    return aprofile, aprofile_target


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
    loggers=None,
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

    for data_iter_step, batch in data_loader:  
        # logging.info("data_iter_step: {}".format(data_iter_step))
        # logging.info("start getting batch")
        sample_track, peak_seq, sample_metadata, celltype_peaks, sample_track_boundary, sample_peak_sequence_boundary, chunk_size, mask, n_peaks, max_n_peaks, total_peak_len, motif_mean_std, labels_data, other_labels, hic_matrix = batch
        if min(chunk_size)<0:
            continue
        # logging.info("Got batch")
        step = data_iter_step // update_freq
        
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if data_iter_step % update_freq == 0:
            apply_schedule_to_optimizer(optimizer, lr_schedule_values, wd_schedule_values, it)

        # logging.info("start cuda computing")
        sample_track = sample_track.to(device, non_blocking=True).bfloat16()
        peak_seq = peak_seq.to(device, non_blocking=True).bfloat16()
        motif_mean_std = motif_mean_std.to(device, non_blocking=True).bfloat16()
        # chunk_size = chunk_size.to(device, non_blocking=True)
        n_peaks = n_peaks.to(device, non_blocking=True)
        labels_data = labels_data.to(device, non_blocking=True).bfloat16()
        other_labels = other_labels.to(device, non_blocking=True).bfloat16()
        atpm = other_labels[:,:,0]
        hic_matrix = hic_matrix.to(device, non_blocking=True).bfloat16()
        result = train_class_batch(model, peak_seq, sample_track, mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, atpm, labels_data, other_labels, criterion, hic_matrix=hic_matrix, args=args)
        loss = result['loss']

        loss_value = check_loss_value(loss)

        grad_norm, loss_scale_value = update_model(model, optimizer, loss_scaler, max_norm, model_ema, update_freq, data_iter_step, loss)

        torch.cuda.synchronize()

        
        # NOTE: evaluation
        # EMA
        test_stats_dict = None
        if (data_iter_step + 1) % update_freq == 0 and args.eval_each_step:
            test_stats_dict = evaluate_all(data_loader_val, model, device, criterion, args, printlog=False)
            
        min_lr, max_lr = get_optimizer_lr_range(optimizer)
        weight_decay_value = get_optimizer_weight_decay(optimizer)

        stats_dict = {
            "loss": loss_value,
            "loss_scale": loss_scale_value,
            "lr": max_lr,
            "min_lr": min_lr,
            "weight_decay": weight_decay_value,
            "grad_norm": grad_norm,
            "epoch": epoch
        }
        if loggers is not None:
            loggers.update(stats_dict, step=it)
        if test_stats_dict is not None:
            loggers.update(test_stats_dict, step=it)
        
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def update_model(model, optimizer, loss_scaler, max_norm, model_ema, update_freq, data_iter_step, loss):
    if loss_scaler is None:
        loss /= update_freq
        model.backward(loss)
        model.step()
        if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
            if model_ema is not None:
                model_ema.update(model)
        grad_norm, loss_scale_value = None, get_loss_scale_for_deepspeed(model)
    else:
        loss /= update_freq
        grad_norm, loss_scale_value = calculate_loss_scale_and_get_grad_norm(
                model,
                optimizer,
                loss_scaler,
                max_norm,
                loss
            )
        if (data_iter_step + 1) % update_freq == 0:
            optimizer.zero_grad()
            if model_ema is not None:
                model_ema.update(model)
    return grad_norm,loss_scale_value


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

    return r2score, pearsonr_score, spearmanr_score


@torch.no_grad()
def evaluate_pretrain(data_loader, model, device, args, epoch=0, printlog=True):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()
    output_masked_list = []
    target_list = []
    for i, batch in enumerate(data_loader):
        sample_track, peak_seq, sample_metadata, celltype_peaks, sample_track_boundary, sample_peak_sequence_boundary, chunk_size, mask, n_peaks, max_n_peaks, total_peak_len, motif_mean_std, _, _, _ = batch
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
    preds_atpm = []
    obs_atpm = []
    preds_aprofile = []
    obs_aprofile = []

    from tqdm import tqdm
    for batch in tqdm(data_loader):
        sample_track, peak_seq, sample_metadata, celltype_peaks, sample_track_boundary, sample_peak_sequence_boundary, chunk_size, mask, n_peaks, max_n_peaks, total_peak_len, motif_mean_std, labels_data, other_labels, hic_matrix = batch
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
        atac_targets = other_labels[:,:,0]
        result = train_class_batch(
            model=model, 
            peak_seq=peak_seq, 
            sample_track=sample_track,
            mask=mask,
            chunk_size=chunk_size,
            n_peaks=n_peaks,
            max_n_peaks=max_n_peaks,
            motif_mean_std=motif_mean_std,
            atac_target=atac_targets,
            exp_target=labels_data,
            other_labels=other_labels,
            criterion=criterion,
            hic_matrix=hic_matrix,
            args=args
        )
        if 'ChrombpNet' in model._get_name()=='GETFinetuneChrombpNet' or 'chrombpnet' in args.model:
            loss = result['loss'].item()
            loss_atpm = result['loss_atpm'].item()
            loss_aprofile = result['loss_aprofile'].item()
            atpm = result['atpm_pred']
            atpm_target = result['atpm_target']
            aprofile = result['aprofile_pred']
            aprofile_target = result['aprofile_target']
            preds_atpm.append(atpm.float().reshape(-1).detach().cpu().numpy())
            obs_atpm.append(atpm_target.float().reshape(-1).detach().cpu().numpy())
            preds_aprofile.append(aprofile.float().reshape(-1).detach().cpu().numpy())
            obs_aprofile.append(aprofile_target.float().reshape(-1).detach().cpu().numpy())

            metric_logger.update(loss=loss)
            metric_logger.update(loss_atpm=loss_atpm)
            metric_logger.update(loss_aprofile=loss_aprofile)

        else:
            exp = result['exp_pred']
            exp_target = result['exp_target']
            atac = result['atac_pred']
            atac_target = result['atac_target']
            loss = result['loss'].item()
            loss_atac = result['loss_atac'].item()
            loss_exp = result['loss_exp'].item()
            loss_confidence = result['loss_confidence'].item()
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
            preds_atac.append(atac.reshape(-1).detach().cpu().numpy())
            obs_atac.append(atac_target.reshape(-1).detach().cpu().numpy())

            metric_logger.update(loss=loss)
            metric_logger.update(loss_atac=loss_atac)
            metric_logger.update(loss_exp=loss_exp)
            metric_logger.update(loss_confidence=loss_confidence)

    if 'ChrombpNet' in model._get_name()=='GETFinetuneChrombpNet' or 'chrombpnet' in args.model:
        preds_atpm = np.concatenate(preds_atpm, axis=0).reshape(-1)
        obs_atpm = np.concatenate(obs_atpm, axis=0).reshape(-1)
        preds_aprofile = np.concatenate(preds_aprofile, axis=0).reshape(-1)
        obs_aprofile = np.concatenate(obs_aprofile, axis=0).reshape(-1)
        bin=100
        obs_aprofile = np.array([np.mean(obs_aprofile[i:i+bin]) for i in range(0, len(obs_aprofile), bin)])
        preds_aprofile = np.array([np.mean(preds_aprofile[i:i+bin]) for i in range(0, len(preds_aprofile), bin)])
        r2score_atpm, pearsonr_score_atpm, spearmanr_score_atpm = cal_score_stats(preds_atpm, obs_atpm, data_loader, args)
        r2score_aprofile, pearsonr_score_aprofile, spearmanr_score_aprofile = cal_score_stats(preds_aprofile, obs_aprofile, data_loader, args)

        metric_logger.meters["r2score"].update(r2score_atpm, n=1)
        metric_logger.meters["pearsonr_score"].update(pearsonr_score_atpm, n=1)    
        metric_logger.meters["spearmanr_score"].update(spearmanr_score_atpm, n=1)
        metric_logger.meters["r2score_atac"].update(r2score_aprofile, n=1)
        metric_logger.meters["pearsonr_score_atac"].update(pearsonr_score_aprofile, n=1)
        metric_logger.meters["spearmanr_score_atac"].update(spearmanr_score_aprofile, n=1)

        if printlog:
            print(
                "* Score@R2 {r2:.3f} Score@pearsonr {pearson:.3f} Score@spearmanr {spearman:.3f}  loss_atpm {loss_atpm:.3f}\n Score@R2 {r2score_aprofile:.3f} Score@pearsonr {pearsonr_score_aprofile:.3f} Score@spearmanr {spearmanr_score_aprofile:.3f}  loss_aprofile {loss_aprofile:.3f}".format(
                    r2=r2score_atpm,
                    pearson=pearsonr_score_atpm,
                    spearman=spearmanr_score_atpm,
                    r2score_aprofile=r2score_aprofile,
                    pearsonr_score_aprofile=pearsonr_score_aprofile,
                    spearmanr_score_aprofile=spearmanr_score_aprofile,
                    loss_aprofile=loss_aprofile,
                    loss_atpm=loss_atpm,
                )
            )

    else:
        preds = np.concatenate(preds, axis=0).reshape(-1)
        obs = np.concatenate(obs, axis=0).reshape(-1)
        preds_atac = np.concatenate(preds_atac, axis=0).reshape(-1)
        obs_atac = np.concatenate(obs_atac, axis=0).reshape(-1)

        r2score, pearsonr_score, spearmanr_score = cal_score_stats(preds, obs, data_loader, args)
        r2score_atac, pearsonr_score_atac, spearmanr_score_atac = cal_score_stats(preds_atac, obs_atac, data_loader, args)

        metric_logger.meters["r2score"].update(r2score, n=1)
        metric_logger.meters["pearsonr_score"].update(pearsonr_score, n=1)
        metric_logger.meters["spearmanr_score"].update(spearmanr_score, n=1)

        metric_logger.meters["r2score_atac"].update(r2score_atac, n=1)
        metric_logger.meters["pearsonr_score_atac"].update(pearsonr_score_atac, n=1)
        metric_logger.meters["spearmanr_score_atac"].update(spearmanr_score_atac, n=1)

        if printlog:
            print(
                "* Score@R2 {r2:.3f} Score@pearsonr {pearson:.3f} Score@spearmanr {spearman:.3f}  loss {losses.global_avg:.3f}".format(
                    r2=r2score,
                    pearson=pearsonr_score,
                    spearman=spearmanr_score,
                    r2_atac=r2score_atac,
                    pearson_atac=pearsonr_score_atac,
                    spearman_atac=spearmanr_score_atac,
                    losses=metric_logger.loss,
                )
            )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
