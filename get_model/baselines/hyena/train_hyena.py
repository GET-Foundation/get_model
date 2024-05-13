import torch
import torch.optim as optim
import json
import os
import subprocess
import transformers
from transformers import PreTrainedModel, AutoModelForCausalLM, PretrainedConfig
from torch.utils.data import DataLoader
from torch import nn
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.sparse import load_npz
import wandb

from atac_rna_data_processing.io.region import Genome

from standalone_hyenadna import HyenaDNAModel, CharacterTokenizer
from utils import GenomicBenchmarkDataset, FetalErythroblastDataset
from huggingface import HyenaDNAPreTrainedModel


def train(model, device, train_loader, optimizer, epoch, loss_fn, log_interval=10):
    """Training loop."""
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target.squeeze())
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            wandb.log({"epoch": epoch, "train_loss": loss.item()})

def test(model, device, test_loader, epoch, loss_fn, save_dir):
    """Test loop."""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target.squeeze()).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    wandb.log({"epoch": epoch, "test_loss": test_loss, "test_accuracy": 100. * correct / len(test_loader.dataset)})
    torch.save(model, f'{save_dir}/checkpoint-{epoch}.pth')


def run_train(args):

    '''
    Main entry point for training.  Select the dataset name and metadata, as
    well as model and training args, and you're off to the genomic races!

    ### GenomicBenchmarks Metadata
    # there are 8 datasets in this suite, choose 1 at a time, with their corresponding settings
    # name                                num_seqs        num_classes     median len    std
    # dummy_mouse_enhancers_ensembl       1210            2               2381          984.4
    # demo_coding_vs_intergenomic_seqs    100_000         2               200           0
    # demo_human_or_worm                  100_000         2               200           0
    # human_enhancers_cohn                27791           2               500           0
    # human_enhancers_ensembl             154842          2               269           122.6
    # human_ensembl_regulatory            289061          3               401           184.3
    # human_nontata_promoters             36131           2               251           0
    # human_ocr_ensembl                   174756          2               315           108.1

    '''
    # experiment settings:
    num_epochs = args.num_epochs  # ~100 seems fine
    max_length = 500  # max len of sequence of dataset (of what you want)
    use_padding = True
    batch_size = args.batch_size
    learning_rate = args.learning_rate  # good default for Hyena
    rc_aug = True  # reverse complement augmentation
    add_eos = False  # add end of sentence token
    weight_decay = 0.1
    save_dir = f"/pmglocal/alb2281/repos/get_model/get_model/baselines/hyena/checkpoints/{args.wandb_run_name}"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # for fine-tuning, only the 'tiny' model can fit on colab
    pretrained_model_name = 'hyenadna-medium-450k-seqlen'  # use None if training from scratch

    # we need these for the decoder head, if using
    use_head = True
    n_classes = 5

    # you can override with your own backbone config here if you want,
    # otherwise we'll load the HF one by default
    backbone_cfg = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # instantiate the model (pretrained here)
    if pretrained_model_name in ['hyenadna-medium-450k-seqlen']:
        # use the pretrained Huggingface wrapper instead
        model = HyenaDNAPreTrainedModel.from_pretrained(
            './checkpoints',
            pretrained_model_name,
            download=True,
            config=backbone_cfg,
            device=device,
            use_head=use_head,
            n_classes=n_classes,
        )

    # from scratch
    else:
        model = HyenaDNAModel(**backbone_cfg, use_head=use_head, n_classes=n_classes)

    # create tokenizer
    tokenizer = CharacterTokenizer(
        characters=['A', 'C', 'G', 'T', 'N'],  # add DNA characters, N is uncertain
        model_max_length=max_length + 2,  # to account for special tokens, like EOS
        add_special_tokens=False,  # we handle special tokens elsewhere
        padding_side='left', # since HyenaDNA is causal, we pad on the left
    )

    if args.split_data:
        genome_path = os.path.join("/manitou/pmg/users/xf2217/interpret_natac/", "hg38.fa")
        genome = Genome('hg38', genome_path)
        atac = pd.read_csv(f"{args.data_dir}/155.atac.bed", sep="\t", header=None)
        atac.columns = ["chrom", "start", "end", "atac"]
        atac = atac.sort_values(by="atac")
        atac["sequence"] = atac.apply(lambda x: genome.get_sequence(x["chrom"], x["start"], x["end"]).seq, axis=1)
        atac["quantile"] = pd.qcut(atac["atac"], q=n_classes, labels=np.arange(n_classes))
        X_train, X_test, y_train, y_test = train_test_split(atac[["sequence", "chrom", "start", "end"]], atac[["quantile"]], test_size=0.2, random_state=42, stratify=atac["quantile"])
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        train_df.to_csv(f"{args.data_dir}/train.csv", index=False) 
        test_df.to_csv(f"{args.data_dir}/test.csv", index=False)
    else:
        train_df = pd.read_csv(f"{args.data_dir}/train.csv")   
        test_df = pd.read_csv(f"{args.data_dir}/test.csv")

    # create datasets
    ds_train = FetalErythroblastDataset(
        data = train_df,
        max_length = max_length,
        use_padding = use_padding,
        tokenizer=tokenizer,
        rc_aug=rc_aug,
        add_eos=add_eos,
    )

    ds_test = FetalErythroblastDataset(
        data = test_df,
        max_length = max_length,
        use_padding = use_padding,
        tokenizer=tokenizer,
        rc_aug=rc_aug,
        add_eos=add_eos,
    )

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    # loss function
    loss_fn = nn.CrossEntropyLoss()

    # create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.to(device)

    for epoch in range(num_epochs):
        train(model, device, train_loader, optimizer, epoch, loss_fn)
        test(model, device, test_loader, epoch, loss_fn, save_dir)
        optimizer.step()


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Train Hyena model on fetal erythroblast data')
    parser.add_argument('--split_data', action='store_true', help='Split data into train and test')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing data')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=6e-4, help='Learning rate for training')
    parser.add_argument("--wandb_project_name", type=str, default="get-finetune", help="wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name")
    args = parser.parse_args()


    wandb.login()
    run = wandb.init(
        project=args.wandb_project_name,
        name=args.wandb_run_name,
        entity="get-v3",
    )
    run_train(args)
