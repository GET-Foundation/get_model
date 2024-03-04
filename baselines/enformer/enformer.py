import argparse
import numpy as np
import os
import pandas as pd
from pathlib import Path
import polars as pl
from scipy.sparse import load_npz
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, TensorDataset
import wandb

from enformer_pytorch import from_pretrained, Enformer, GenomeIntervalDataset
from enformer_pytorch.finetune import HeadAdapterWrapper
from enformer_pytorch.data import FastaInterval, identity

from get_model.metrics import score_r2, score_spearmanr, score_pearsonr
from get_model.utils import get_test_stats


hg38_path = "/pmglocal/alb2281/get_data/get_resources/hg38.ml.fa"
atac_data = "/pmglocal/alb2281/get_data/k562_count_10/k562_count_10.csv"
labels_path = "/pmglocal/alb2281/get_data/k562_count_10/k562_count_10.watac.npz"
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


def get_eval_stats(preds, targets):
    r2score = score_r2(preds, targets)
    spearmanr_score = score_spearmanr(preds, targets)
    pearsonr_score = score_pearsonr(preds, targets)
    ret_scores = {
        "val_r2": r2score,
        "val_spearmanr": spearmanr_score,
        "val_pearsonr": pearsonr_score
    }
    return ret_scores


def poisson_loss(pred, target):
    return (pred - target * log(pred)).mean()


def train_enformer(args):
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
    cudnn.benchmark = True

    base_dir = f"/pmglocal/alb2281/get_data/k562_count_10/splits/{args.leaveout_chr}"

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        train_path, val_path = split_data_by_chr(base_dir, args.leaveout_chr)
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

    # model_ema = None
    # if args.model_ema:
    #     # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
    #     model_ema = ModelEma(
    #         model,
    #         decay=args.model_ema_decay,
    #         device="cpu" if args.model_ema_force_cpu else "",
    #         resume="",
    #     )
    #     print("Using EMA with decay = %.8f" % args.model_ema_decay)

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


            
            
            running_loss += loss
            loss.backward()

            if (idx + 1) % args.accumulation_steps == 0:  # Update every accumulation_steps
                optimizer.step()
                optimizer.zero_grad()

            if (train_steps + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.num_epochs}], Step [{train_steps+1}/{len(train_loader)}], Train Loss: {running_loss.item() / (idx+1):.4f}")
            train_steps += 1
            

        avg_loss = running_loss.item() / len(train_loader)
        train_stats = {
            "train_epoch": epoch,
            "train_loss": avg_loss,
        }
        # wandb.log(train_stats)
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Step [{train_steps+1}/{len(train_loader)}], Train Loss: {avg_loss:.4f}")

        if args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0:
            val_loss = 0.0
            for idx, (seq_batch, target_batch) in enumerate(val_loader):
                target_batch = target_batch.unsqueeze(1)
                target_batch = target_batch.unsqueeze(2)
                seq_batch = seq_batch.to(device)
                target_batch = target_batch.to(device)

                preds = enformer(seq_batch)
                loss = poisson_loss(preds, target_batch)
                val_loss += loss
            
            avg_val_loss = val_loss.item() / len(val_loader)
            test_stats = get_test_stats(preds, target_batch)
            test_stats += {
                "val_loss": avg_val_loss,
            }
            # wandb.log(test_stats)
            print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}")
            

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", type=int, default=2)
    parser.add_argument("-w", "--num_workers", type=int, default=4)
    parser.add_argument("-e", "--num_epochs", type=int, default=10)
    parser.add_argument("-a", "--accumulation_steps", type=int, default=1)
    parser.add_argument("-l", "--learning_rate", type=float, default=1e-5)
    parser.add_argument("--eval_freq", type=int, default=1, help="Frequency of evaluation")
    parser.add_argument("--wandb_project_name", type=str, default="enformer-baseline", help="Wandb project name")
    parser.add_argument("--wandb_entity_name", type=str, default="get-v3", help="Wandb entity name")
    parser.add_argument("--wandb_run_name", type=str, default="None", help="Wandb entity name")
    parser.add_argument("--leaveout_chr", type=str, default="chr11")

    args = parser.parse_args()
    train_enformer(args)
