import math
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn

from get_model.model.motif import parse_meme_file


def dict_to_device(dict, device):
    for key, value in dict.items():
        if isinstance(value, torch.Tensor):
            dict[key] = value.to(device)
    return dict


@dataclass
class BaseConfig:
    """Dummy configuration class with arbitrary generated values.

    Args:
        freezed (bool|str): Whether to freeze the parameters. If True, all parameters are freezed.
            If a string, only parameters with the string in their name are freezed. Defaults to False."""
    _target_: str = "get_model.model.modules.BaseConfig"
    freezed: bool | str = False


class BaseModule(nn.Module):
    """Base model class with methods to generate dummy data and forward function.

    Args:
        cfg (BaseConfig): Configuration object.
    """

    def __init__(self, cfg: BaseConfig):
        super().__init__()
        self.cfg = cfg

    def generate_dummy_data(self, batch_size=1):
        """Generate dummy data for the forward function."""
        raise NotImplementedError(
            "Dummy data generation must be implemented in subclasses.")

    def forward(self, x):
        """Dummy forward function."""
        raise NotImplementedError(
            "Forward function must be implemented in subclasses.")

    def test(self, device='cpu'):
        """Test the forward function with dummy data."""
        x = self.generate_dummy_data()
        self.to(device)
        x = dict_to_device(x, device)
        return self(**x)

    def freeze_parameters(self):
        """Freeze module parameters based on the configuration."""
        if self.cfg.freezed:
            if self.cfg.freezed == True:
                for param in self.parameters():
                    param.requires_grad = False
            else:
                for name, param in self.named_parameters():
                    if self.cfg.freezed in name:
                        param.requires_grad = False


@dataclass
class RegionEmbedConfig(BaseConfig):
    """Configuration class for the region embedding module.

    Args:
        num_features (int): Number of features.
        embed_dim (int): Dimension of the embedding."""
    _target_: str = "get_model.model.modules.RegionEmbedConfig"
    num_features: int = 800
    embed_dim: int = 768


class RegionEmbed(BaseModule):
    """A simple region embedding transforming motif features to region embeddings.
    """

    def __init__(self, cfg: RegionEmbedConfig):
        super().__init__(cfg)
        self.embed = nn.Linear(cfg.num_features, cfg.embed_dim)

    def forward(self, x, **kwargs):
        x = self.embed(x)
        return x

    def generate_dummy_data(self, batch_size=1):
        return torch.rand(batch_size, 5, self.cfg.num_features)


@dataclass
class MotifScannerConfig(BaseConfig):
    """
    Configuration class for the motif scanner module.

    Args:
        num_motif (int): Number of motifs to scan.
        include_reverse_complement (bool): Whether to include reverse complement motifs.
        bidirectional_except_ctcf (bool): Whether to include reverse complement motifs for all motifs except CTCF.
        motif_prior (bool): Whether to use motif prior.
        learnable (bool): Whether to make the motif scanner learnable."""
    _target_: str = "get_model.model.modules.MotifScannerConfig"
    num_motif: int = 637
    include_reverse_complement: bool = True
    bidirectional_except_ctcf: bool = False
    motif_prior: bool = True
    learnable: bool = False


class MotifScanner(BaseModule):
    """A motif encoder based on Conv1D, supporting initialized with PWM prior.

    Architecture:
    Conv1D(4, self.num_kernel, 29, padding='same', activation='relu')

    Args:
        num_motif (int): Number of motifs to scan.
        include_reverse_complement (bool): Whether to include reverse complement motifs.
        bidirectional_except_ctcf (bool): Whether to include reverse complement motifs for all motifs except CTCF.
        motif_prior (bool): Whether to use motif prior.
        learnable (bool): Whether to make the motif scanner learnable.
    """

    def __init__(self, cfg: MotifScannerConfig):
        super().__init__(cfg)
        self.num_kernel = cfg.num_motif
        self.bidirectional_except_ctcf = cfg.bidirectional_except_ctcf
        self.motif_prior = cfg.motif_prior
        self.learnable = cfg.learnable
        if cfg.include_reverse_complement and self.bidirectional_except_ctcf:
            self.num_kernel *= 2
        elif cfg.include_reverse_complement:
            self.num_kernel *= 2

        motifs = self.load_pwm_as_kernel(
            include_reverse_complement=cfg.include_reverse_complement)
        self.motif = nn.Sequential(
            nn.Conv1d(4, self.num_kernel, 29, padding="same", bias=True),
            # nn.BatchNorm1d(num_motif),
            nn.ReLU(),
        )
        if self.motif_prior:
            assert (
                motifs.shape == self.motif[0].weight.shape
            ), f"Motif prior shape ({motifs.shape}) doesn't match model ({self.motif[0].weight.shape})."
            self.motif[0].weight.data = motifs
        if not self.learnable:
            self.motif[0].weight.requires_grad = False

    def load_pwm_as_kernel(
        self,
        pwm_path="https://resources.altius.org/~jvierstra/projects/motif-clustering-v2.1beta/consensus_pwms.meme",
        include_reverse_complement=True,
    ):
        # download pwm to local
        if not os.path.exists("consensus_pwms.meme"):
            os.system(f"wget {pwm_path}")
        # load pwm
        motifs = parse_meme_file("consensus_pwms.meme")
        motifs_rev = motifs[:, ::-1, ::-1].copy()
        # construct reverse complement
        motifs = torch.tensor(motifs)
        motifs_rev = torch.tensor(motifs_rev)
        motifs = torch.cat([motifs, motifs_rev], dim=0)
        return motifs.permute(0, 2, 1).float()

    def normalize_motif(self, x, motif_mean_std):
        return (x - motif_mean_std[:, 0, :].unsqueeze(1)) / motif_mean_std[:, 1, :].unsqueeze(1)

    def forward(self, x, motif_mean_std=None):
        x = x.permute(0, 2, 1)
        x = self.motif(x)
        if self.bidirectional_except_ctcf:
            # get ctcf scanned score for both motif and reverse complement motif. idx of ctcf is 77 and 637+77=714
            ctcf = x[:, 77, :]
            ctcf_rev = x[:, 714, :]
            # combine motif and reverse complement motif for all motifs
            x = x[:, :637, :] + x[:, 637:, :]
            # add ctcf/ctcf_rev score to the end
            x = torch.cat([x, ctcf.unsqueeze(1), ctcf_rev.unsqueeze(1)], dim=1)
        x = x.permute(0, 2, 1)
        if motif_mean_std is not None:
            x = self.normalize_motif(x, motif_mean_std)
        x = F.relu(x)
        return x

    def generate_dummy_data(self, batch_size=1):
        return torch.rand(batch_size, 5, 4, 1000)

    def cuda(self, device=None):
        self.motif = self.motif.cuda()
        return self._apply(lambda t: t.cuda(device))


@dataclass
class ExpressionHeadConfig(BaseConfig):
    """Configuration class for the expression head.

    Args:
        embed_dim (int): Dimension of the embedding.
        output_dim (int): Dimension of the output."""
    embed_dim: int = 768
    output_dim: int = 2
    use_atac: bool = False


class ExpressionHead(BaseModule):
    """Expression head"""

    def __init__(self, cfg: ExpressionHeadConfig):
        super().__init__(cfg)
        self.use_atac = cfg.use_atac
        if self.use_atac:
            self.head = nn.Linear(cfg.embed_dim + 1, cfg.output_dim)
        else:
            self.head = nn.Linear(cfg.embed_dim, cfg.output_dim)

    def forward(self, x, atac=None):
        if self.use_atac:
            x = torch.cat([x, atac], dim=-1)
        return self.head(x)


@dataclass
class ATACHeadConfig(BaseConfig):
    """Configuration class for the ATAC head.

    Args:
        embed_dim (int): Dimension of the embedding.
        hidden_dim (int): Dimension of the hidden layer.
        output_dim (int): Dimension of the output.
        drop (float): Dropout rate. Defaults to 0.1."""
    _target_: str = "get_model.model.modules.ATACHeadConfig"
    embed_dim: int = 768
    hidden_dim: int = 256
    output_dim: int = 1
    drop: float = 0.1


class ATACHead(BaseModule):
    """ATAC head"""

    def __init__(self, cfg: ATACHeadConfig):
        super().__init__(cfg)
        self.fc1 = nn.Linear(cfg.embed_dim, cfg.hidden_dim)
        self.fc2 = nn.Linear(cfg.hidden_dim, cfg.output_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(cfg.drop)
        self.drop2 = nn.Dropout(cfg.drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def pool(x, method='mean'):
    """
    x: (L,D)
    """
    if x.shape[0] == 0:
        return torch.zeros(x.shape[1], dtype=x.dtype, device=x.device)
    if method == 'sum':
        return x.sum(0)
    elif method == 'max':
        return x.max(0)
    elif method == 'mean':
        return x.mean(0)
    else:
        raise ValueError(f"Invalid pool_method: {method}")


@dataclass
class SplitPoolConfig(BaseConfig):
    """Configuration class for the SplitPool module.

    Args:
        pool_method (str): The method to pool the tensor. Defaults to 'mean'.
    """
    _target_: str = "get_model.model.modules.SplitPoolConfig"
    pool_method: str = 'mean'


class SplitPool(BaseModule):
    """
    Receive a tensor of shape (batch, length, dimension) and split along length
    dimension based on a celltype_peak tensor of shape (batch, n_peak, 2) where
    the second dimension is the start and end of the peak. The length dimension
    can be decomposed into a sum of the peak lengths with each peak padded left
    and right with 50bp and directly concatenated. Thus the boundary for the
    splitting can be calculated by cumsum of the padded peak lengths. The output
    is a tensor of shape (batch, n_peak, dimension).
    """

    def __init__(self, cfg: SplitPoolConfig):
        super().__init__(cfg)
        self.pool_method = cfg.pool_method

    def forward(self,
                x: torch.Tensor,
                chunk_size: int | List[int],
                n_peaks: torch.Tensor,
                max_n_peaks: int):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch, length, dimension).
            chunk_size (int | List[int]): The size of each chunk to pool.
            n_peaks (torch.Tensor): The number of peaks for each sample.
            max_n_peaks (int): The maximum number of peaks in the batch.

        Returns:
            torch.Tensor: Output tensor of shape (batch, n_peak, dimension).
        """
        batch, length, embed_dim = x.shape
        chunk_list = torch.split(x.reshape(-1, embed_dim), chunk_size, dim=0)
        chunk_list = torch.vstack([pool(chunk) for chunk in chunk_list])

        pool_idx = torch.cumsum(n_peaks + 1, 0)
        pool_left_pad = torch.tensor(0).unsqueeze(0).to(pool_idx.device)
        pool_start = torch.cat([pool_left_pad, pool_idx[:-1]])
        pool_end = pool_idx - 1

        pool_list = [chunk_list[pool_start[i]:pool_end[i]]
                     for i in range(len(pool_start))]
        x = torch.stack([
            torch.cat([
                pool_list[i],
                torch.zeros(
                    max_n_peaks - n_peaks[i], embed_dim).to(pool_list[i].device)
            ])
            for i in range(len(pool_list))
        ])

        return x

    def pool(self, chunk):
        """
        Pool the tensor using the specified pool_method.

        Args:
            chunk (torch.Tensor): Input tensor to be pooled.

        Returns:
            torch.Tensor: Pooled tensor.
        """
        if self.pool_method == 'mean':
            return torch.mean(chunk, dim=0)
        elif self.pool_method == 'max':
            return torch.max(chunk, dim=0).values
        else:
            raise ValueError(f"Invalid pool_method: {self.pool_method}")

    def generate_dummy_data(self, batch_size=1):
        batch, num_peaks_per_sample, peak_len = 2, 10, 500
        total_length = num_peaks_per_sample * peak_len
        embed_dim = 100
        n_peaks = torch.tensor([10, 10])
        max_n_peaks = torch.max(n_peaks).item()
        chunk_size = [peak_len] * batch_size * max_n_peaks
        x = torch.rand(batch_size, total_length, embed_dim)
        return x, chunk_size, n_peaks, max_n_peaks


def exponential_linspace_int(start, end, num, divisible_by=1):
    def _round(x):
        return int(round(x / divisible_by) * divisible_by)

    base = math.exp(math.log(end / start) / (num - 1))
    return [_round(start * base**i) for i in range(num)]


@dataclass
class ResidualConfig(BaseConfig):
    """Configuration class for the Residual module."""
    _target_: str = "get_model.model.modules.ResidualConfig"
    pass


class Residual(BaseModule):
    def __init__(self, cfg: ResidualConfig, fn: nn.Module):
        super().__init__(cfg)
        self.fn = fn

    def forward(self, x, **kwargs):
        """
        Args:
            x (torch.Tensor): Input tensor.
            **kwargs: Additional keyword arguments passed to the inner function.

        Returns:
            torch.Tensor: Output tensor with the residual connection.
        """
        return self.fn(x, **kwargs) + x

    def generate_dummy_data(self, batch_size=1):
        # Implement the dummy data generation based on the inner function
        raise NotImplementedError(
            "Dummy data generation for Residual module is not implemented.")


@dataclass
class DilatedConv1dConfig(BaseConfig):
    """Configuration class for the DilatedConv1d module.

    Args:
        dim (int): Number of input and output channels.
        kernel_size (int): Size of the convolutional kernel. Defaults to 3.
        dilation (int): Dilation factor for the convolution. Defaults to 1.
    """
    _target_: str = "get_model.model.modules.DilatedConv1dConfig"
    dim: int = 64
    kernel_size: int = 3
    dilation: int = 1


class DilatedConv1d(BaseModule):
    """A dilated 1D convolution with gated activation units."""

    def __init__(self, cfg: DilatedConv1dConfig):
        super().__init__(cfg)
        self.conv = nn.Conv1d(cfg.dim, cfg.dim, cfg.kernel_size,
                              padding='valid', dilation=cfg.dilation)
        self.activation = nn.ReLU()
        # self.batch_norm = nn.BatchNorm1d(cfg.dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch, dim, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch, dim, length).
        """
        out = self.conv(x)
        # out = self.batch_norm(out)
        diff_length = x.shape[-1] - out.shape[-1]
        assert (diff_length % 2 == 0)
        crop_size = diff_length // 2
        if crop_size > 0:
            x = x[..., crop_size:-crop_size]
        return self.activation(out) + x

    def generate_dummy_data(self, batch_size=1):
        length = 1000  # Adjust the length as needed
        x = torch.rand(batch_size, self.cfg.dim, length)
        return x


@dataclass
class DilatedConv1dBlockConfig(BaseConfig):
    """Configuration class for the DilatedConv1dBlock module.

    Args:
        dim (int): Number of input and output channels.
        depth (int): Number of dilated convolution layers. Defaults to 3.
        kernel_size (int): Size of the convolutional kernel. Defaults to 3.
    """
    _target_: str = "get_model.model.modules.DilatedConv1dBlockConfig"
    hidden_dim: int = 64
    depth: int = 7
    kernel_size: int = 3


class DilatedConv1dBlock(BaseModule):
    """A series of dilated 1D convolutions with expanding dilation by a factor of 2, starting from 4"""

    def __init__(self, cfg: DilatedConv1dBlockConfig):
        super().__init__(cfg)
        self.layers = nn.ModuleList([])
        for i in range(cfg.depth):
            dilation = 2 ** (i + 1)
            self.layers.append(DilatedConv1d(DilatedConv1dConfig(
                dim=cfg.hidden_dim, kernel_size=cfg.kernel_size, dilation=dilation)))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch, dim, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch, dim, length).
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def generate_dummy_data(self, batch_size=1):
        length = 1000  # Adjust the length as needed
        x = torch.rand(batch_size, self.cfg.dim, length)
        return x


@dataclass
class ConvPoolConfig(BaseConfig):
    """Configuration class for the ConvPool module.

    Args:
        motif_dim (int): Dimension of the motif input.
        hidden_dim (int): Dimension of the hidden layer.
        n_dil_layers (int): Number of dilated convolution layers. Defaults to 7.
        profile_kernel_size (int): Kernel size for the aprofile header. Defaults to 75.
        pool_method (str): Method to pool the tensor. Defaults to 'mean'.
    """
    _target_: str = "get_model.model.modules.ConvPoolConfig"
    pool_method: str = 'mean'
    motif_dim: int = 639
    hidden_dim: int = 256
    n_dil_layers: int = 7
    profile_kernel_size: int = 75


class ConvPool(BaseModule):
    """
    Receive a tensor of shape (batch, length, dimension) and split along length
    dimension based on a celltype_peak tensor of shape (batch, n_peak, 2) where
    the second dimension is the start and end of the peak. The length dimension
    can be decomposed into a sum of the peak lengths with each peak padded left
    and right with 50bp and directly concatenated. Thus the boundary for the
    splitting can be calculated by cumsum of the padded peak lengths. The output
    is a tensor of shape (batch, n_peak, dimension).
    """

    def __init__(self, cfg: ConvPoolConfig):
        super().__init__(cfg)
        self.pool_method = cfg.pool_method
        # self.motif_proj = nn.Linear(
        #     cfg.motif_dim, cfg.hidden_dim)
        self.dila_conv_tower = DilatedConv1dBlock(DilatedConv1dBlockConfig(
            hidden_dim=cfg.hidden_dim, depth=cfg.n_dil_layers))
        self.aprofile_header = nn.Conv1d(
            cfg.hidden_dim, 1, cfg.profile_kernel_size, padding='valid')
        self.atpm_header = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, x, peak_split, n_peaks, max_n_peaks):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch, length, dimension).
            peak_split (torch.Tensor): Tensor of shape (num_peaks,) indicating the size of each peak.
            n_peaks (torch.Tensor): Tensor of shape (batch,) indicating the number of peaks for each sample.
            max_n_peaks (int): Maximum number of peaks in the batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - peak_atpms: Tensor of shape (batch, max_n_peaks, 1) representing the aTPM values for each peak.
                - peak_profiles: Tensor of shape (batch, length) representing the peak profiles.
        """
        # x = self.motif_proj(x)
        batch, length, motif_dim = x.shape

        if torch.all(n_peaks == max_n_peaks) and np.unique(peak_split).shape[0] == 2:
            peak_size = np.unique(peak_split).max()
            num_peaks = n_peaks[0]
            x = x.reshape(batch * num_peaks, peak_size,
                          motif_dim).transpose(1, 2)
            peak_profile = self.dila_conv_tower(x)
            peak_atpm = peak_profile.mean(2)
            peak_profile = self.aprofile_header(peak_profile)
            peak_atpm = self.atpm_header(peak_atpm)
            peak_atpm = peak_atpm.reshape(batch, num_peaks)
            peak_profile = peak_profile.reshape(batch, -1)
            peak_profile = peak_profile
            return peak_atpm, peak_profile
        else:
            peak_list = torch.split(
                x.reshape(-1, motif_dim), peak_split, dim=0)
            peak_atpm_list = []
            peak_profiles = []
            for peak in peak_list:
                if peak.shape[0] == 0:
                    peak_profiles.append(torch.zeros(0).to(peak.device))
                    continue
                peak_profile = self.dila_conv_tower(
                    peak.unsqueeze(0).transpose(1, 2))
                peak_atpm = peak_profile.mean(2)
                peak_profile = self.aprofile_header(peak_profile).squeeze(0)
                peak_atpm_list.append(peak_atpm)
                peak_profiles.append(peak_profile)
            peak_atpm_list = torch.vstack(peak_atpm_list)
            peak_profiles = torch.cat(peak_profiles).view(batch, length)
            peak_atpm = self.atpm_header(peak_atpm_list)

            pool_idx = torch.cumsum(n_peaks + 1, 0)
            pool_left_pad = torch.tensor(0).unsqueeze(0).to(pool_idx.device)
            pool_start = torch.cat([pool_left_pad, pool_idx[:-1]])
            pool_end = pool_idx - 1
            pool_list = [peak_atpm[pool_start[i]:pool_end[i]]
                         for i in range(len(pool_start))]
            peak_atpms = torch.stack([torch.cat([pool_list[i], torch.zeros(
                max_n_peaks - n_peaks[i], 1).to(pool_list[i].device)]) for i in range(len(pool_list))])
            return peak_atpms, peak_profiles

    def generate_dummy_data(self, batch_size=1):
        length = 1000
        motif_dim = self.cfg.motif_dim
        max_n_peaks = 100
        n_peaks = torch.randint(10, max_n_peaks, (batch_size,))
        peak_split = torch.randint(50, 100, (max_n_peaks,))
        x = torch.rand(batch_size, length, motif_dim)
        return x, peak_split, n_peaks, max_n_peaks


@dataclass
class ATACSplitPoolConfig(BaseConfig):
    """Configuration class for the ATACSplitPool module.

    Args:
        pool_method (str): The method to pool the tensor. Defaults to 'mean'.
        atac_kernel_num (int): Number of kernels for ATAC convolution. Defaults to 16.
        motif_dim (int): Dimension of the motif input. Defaults to 639.
        joint_kernel_num (int): Number of kernels for joint convolution. Defaults to 16.
        atac_kernel_size (int): Kernel size for ATAC convolution. Defaults to 3.
        joint_kernel_size (int): Kernel size for joint convolution. Defaults to 3.
        final_bn (bool): Whether to apply batch normalization at the end. Defaults to False.
        binary_atac (bool): Whether to binarize the ATAC signal. Defaults to False.
    """
    _target_: str = "get_model.model.modules.ATACSplitPoolConfig"
    pool_method: str = 'mean'
    motif_dim: int = 639
    atac_kernel_num: int = 161
    joint_kernel_num: int = 161
    atac_kernel_size: int = 3
    joint_kernel_size: int = 3
    final_bn: bool = False
    binary_atac: bool = False


class ATACSplitPool(BaseModule):
    """
    Receive a tensor of shape (batch, length, dimension) and split along length
    dimension based on a celltype_peak tensor of shape (batch, n_peak, 2) where
    the second dimension is the start and end of the peak. The length dimension
    can be decomposed into a sum of the peak lengths with each peak padded left
    and right with 50bp and directly concatenated. Thus the boundary for the
    splitting can be calculated by cumsum of the padded peak lengths. The output
    is a tensor of shape (batch, n_peak, dimension).
    """

    def __init__(self, cfg: ATACSplitPoolConfig):
        super().__init__(cfg)
        self.pool_method = cfg.pool_method
        self.atac_conv = nn.Conv1d(
            1, cfg.atac_kernel_num, cfg.atac_kernel_size, padding="same", bias=False)
        self.atac_bn = nn.BatchNorm1d(cfg.atac_kernel_num, affine=False)
        self.joint_conv = nn.Conv1d(cfg.motif_dim + cfg.atac_kernel_num,
                                    cfg.joint_kernel_num, cfg.joint_kernel_size, padding="same", bias=False)
        self.joint_bn = nn.BatchNorm1d(cfg.joint_kernel_num, affine=False)
        self.patch_pool = nn.MaxPool1d(25, stride=25)
        if cfg.final_bn:
            self.final_bn = nn.BatchNorm1d(
                cfg.motif_dim + cfg.joint_kernel_num, affine=False)
        self.binary_atac = cfg.binary_atac

    def forward(self, x, atac, peak_split, n_peaks, max_n_peaks):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch, length, dimension).
            atac (torch.Tensor): ATAC signal tensor of shape (batch, length, 1).
            peak_split (list): List of peak lengths for splitting.
            n_peaks (torch.Tensor): Tensor of shape (batch,) indicating the number of peaks for each sample.
            max_n_peaks (int): Maximum number of peaks in the batch.

        Returns:
            torch.Tensor: Output tensor of shape (batch, n_peak, motif_dim + joint_kernel_num).
        """
        if self.binary_atac:
            atac = (atac > 0).float()
        else:
            atac = torch.log10(atac + 1)

        x_region = self.forward_x(x, peak_split, n_peaks, max_n_peaks)
        joint_region = self.forward_joint(
            x, atac, peak_split, n_peaks, max_n_peaks)

        joint_region = torch.log2(joint_region + 1)
        x = torch.cat([x_region, joint_region], dim=2).contiguous()

        if hasattr(self, 'final_bn'):
            x = self.final_bn(x.transpose(1, 2)).transpose(1, 2)

        return x

    def forward_joint(self, x, atac, peak_split, n_peaks, max_n_peaks, patch_size=25):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch, length, dimension).
            atac (torch.Tensor): ATAC signal tensor of shape (batch, length, 1).
            peak_split (list): List of peak lengths for splitting.
            n_peaks (torch.Tensor): Tensor of shape (batch,) indicating the number of peaks for each sample.
            max_n_peaks (int): Maximum number of peaks in the batch.
            patch_size (int): Size of each patch to pool. Defaults to 25.

        Returns:
            torch.Tensor: Output tensor of shape (batch, n_peak, joint_kernel_num).
        """
        x = x.transpose(1, 2).contiguous()
        atac = atac.unsqueeze(1).contiguous()
        x_pooled = self.patch_pool(x)
        atac_pooled = self.patch_pool(atac)
        patch_peak_split = [i // patch_size for i in peak_split]

        atac_pooled = self.atac_conv(atac_pooled)
        atac_pooled = self.atac_bn(atac_pooled)
        atac_pooled = F.relu(atac_pooled)

        x_pooled = torch.cat([x_pooled, atac_pooled], dim=1).contiguous()
        x_pooled = self.joint_conv(x_pooled)
        x_pooled = self.joint_bn(x_pooled)
        x_pooled = F.relu(x_pooled).transpose(1, 2)
        batch, length, embed_dim = x_pooled.shape

        chunk_list = torch.split(
            x_pooled.reshape(-1, embed_dim), patch_peak_split, dim=0)
        chunk_list = torch.vstack(
            [pool(chunk, self.pool_method) for chunk in chunk_list])

        pool_idx = torch.cumsum(n_peaks + 1, 0)
        pool_left_pad = torch.tensor(0).unsqueeze(0).to(pool_idx.device)
        pool_start = torch.cat([pool_left_pad, pool_idx[:-1]])
        pool_end = pool_idx - 1
        pool_list = [chunk_list[pool_start[i]:pool_end[i]]
                     for i in range(len(pool_start))]
        x_pooled = torch.stack([torch.cat([pool_list[i], torch.zeros(
            max_n_peaks - n_peaks[i], embed_dim).to(pool_list[i].device)]) for i in range(len(pool_list))])

        return x_pooled

    def forward_x(self, x, peak_split, n_peaks, max_n_peaks):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch, length, dimension).
            peak_split (list): List of peak lengths for splitting.
            n_peaks (torch.Tensor): Tensor of shape (batch,) indicating the number of peaks for each sample.
            max_n_peaks (int): Maximum number of peaks in the batch.

        Returns:
            torch.Tensor: Output tensor of shape (batch, n_peak, dimension).
        """
        batch, length, embed_dim = x.shape
        chunk_list = torch.split(x.reshape(-1, embed_dim), peak_split, dim=0)
        chunk_list = torch.vstack(
            [pool(chunk, self.pool_method) for chunk in chunk_list])

        pool_idx = torch.cumsum(n_peaks + 1, 0)
        pool_left_pad = torch.tensor(0).unsqueeze(0).to(pool_idx.device)
        pool_start = torch.cat([pool_left_pad, pool_idx[:-1]])
        pool_end = pool_idx - 1
        pool_list = [chunk_list[pool_start[i]:pool_end[i]]
                     for i in range(len(pool_start))]
        x = torch.stack([torch.cat([pool_list[i], torch.zeros(
            max_n_peaks - n_peaks[i], embed_dim).to(pool_list[i].device)]) for i in range(len(pool_list))])

        return x

    def generate_dummy_data(self, batch_size=1):
        length = 1000
        embed_dim = self.cfg.motif_dim
        max_n_peaks = 100
        n_peaks = torch.randint(10, max_n_peaks, (batch_size,))
        peak_split = torch.randint(50, 100, (max_n_peaks,)).tolist()
        x = torch.rand(batch_size, length, embed_dim)
        atac = torch.rand(batch_size, length, 1)
        return x, atac, peak_split, n_peaks, max_n_peaks


@dataclass
class ATACSplitPoolMaxNormConfig(ATACSplitPoolConfig):
    _target_: str = "get_model.model.modules.ATACSplitPoolMaxNormConfig"
    pass


class ATACSplitPoolMaxNorm(ATACSplitPool):
    """
    Receive a tensor of shape (batch, length, dimension) and split along length
    dimension based on a celltype_peak tensor of shape (batch, n_peak, 2) where
    the second dimension is the start and end of the peak. The length dimension
    can be decomposed into a sum of the peak lengths with each peak padded left
    and right with 50bp and directly concatenated. Thus the boundary for the
    splitting can be calculated by cumsum of the padded peak lengths. The output
    is a tensor of shape (batch, n_peak, dimension).
    """

    def __init__(self, cfg: ATACSplitPoolMaxNormConfig):
        super().__init__(cfg)
        self.register_buffer('running_max', torch.ones(cfg.motif_dim))
        self.register_buffer('running_max_joint',
                             torch.ones(cfg.joint_kernel_num))

    def forward(self, x, atac, peak_split, n_peaks, max_n_peaks):
        if self.binary_atac:
            atac = (atac > 0).float()
        else:
            atac = atac / (atac.max(1, keepdim=True)[0] + 1e-5)

        x_region = self.forward_x(x, peak_split, n_peaks, max_n_peaks)
        joint_region = self.forward_joint(
            x, atac, peak_split, n_peaks, max_n_peaks)

        with torch.no_grad():
            self.running_max = torch.max(self.running_max, torch.max(
                x_region.view(-1, self.cfg.motif_dim), dim=0).values)
            self.running_max_joint = torch.max(self.running_max_joint, torch.max(
                joint_region.view(-1, self.cfg.joint_kernel_num), dim=0).values)

        x_region = x_region / \
            (self.running_max.unsqueeze(0).unsqueeze(0) + 1e-5)
        joint_region = joint_region / \
            (self.running_max_joint.unsqueeze(0).unsqueeze(0) + 1e-5)

        x = torch.cat([x_region, joint_region], dim=2).contiguous()

        if hasattr(self, 'final_bn'):
            x = self.final_bn(x.transpose(1, 2)).transpose(1, 2)

        return x
