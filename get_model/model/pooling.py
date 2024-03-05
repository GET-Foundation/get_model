import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn
import math

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


class SplitPool(nn.Module):
    """
    Receive a tensor of shape (batch, length, dimension) and split along length
    dimension based on a celltype_peak tensor of shape (batch, n_peak, 2) where
    the second dimension is the start and end of the peak. The length dimension 
    can be decomposed into a sum of the peak lengths with each peak padded left 
    and right with 50bp and directly concatenated. Thus the boundary for the 
    splitting can be calculated by cumsum of the padded peak lengths. The output
    is a tensor of shape (batch, n_peak, dimension). 
    """
    def __init__(self, pool_method='mean'):
        super().__init__()
        self.pool_method = pool_method


    def forward(self, x, chunk_size, n_peaks, max_n_peaks):
        """
        x: (batch, length, dimension)
        chunk_size: the size of each chunk to pool        
        n_peaks: the number of peaks for each sample
        max_n_peaks: the maximum number of peaks in the batch
        pool_method: the method to pool the tensor
        """
        batch, length, embed_dim = x.shape
        chunk_list = torch.split(x.reshape(-1,embed_dim), chunk_size, dim=0)
        # each element is L, D, pool the tensor
        chunk_list = torch.vstack([pool(chunk, self.pool_method) for chunk in chunk_list])
        # remove the padded part
        pool_idx = torch.cumsum(n_peaks+1,0)
        pool_left_pad = torch.tensor(0).unsqueeze(0).to(pool_idx.device)
        pool_start = torch.cat([pool_left_pad, pool_idx[:-1]])
        pool_end = pool_idx-1
        pool_list = [chunk_list[pool_start[i]:pool_end[i]] for i in range(len(pool_start))]
        # pad the element in pool_list if the number of peaks is not the same
        x = torch.stack([torch.cat([pool_list[i], torch.zeros(max_n_peaks-n_peaks[i], embed_dim).to(pool_list[i].device)]) for i in range(len(pool_list))])

        return x

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
def exponential_linspace_int(start, end, num, divisible_by = 1):
    def _round(x):
        return int(round(x / divisible_by) * divisible_by)

    base = math.exp(math.log(end / start) / (num - 1))
    return [_round(start * base**i) for i in range(num)]
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class DilatedConv1d(nn.Module):
    """A dilated 1D convolution with gated activation units. Don't change the 
    sequence length. 
    """
    def __init__(self, dim, kernel_size = 3, dilation = 1):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding = (kernel_size - 1) * dilation // 2, dilation = dilation)
        self.activation = nn.GELU()
    def forward(self, x):
        x = self.conv(x)
        return self.activation(x) + x
    
class DilatedConv1dBlock(nn.Module):
    """A series of dilated 1D convolutions with expanding dilation by a factor of 2, starting from 4"""
    def __init__(self, dim, depth = 3, kernel_size = 3):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            dilation = 2 ** (i+1)
            self.layers.append(DilatedConv1d(dim, kernel_size = kernel_size, dilation = dilation))
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x






class SequenceDecoder(nn.Module):
    """A sequence decoder that takes in a sequence of embeddings and outputs a sequence of logits for one-hot prediction"""
    def __init__(self, dim, depth, num_tokens, max_seq_len, kernel_size = 3):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.max_seq_len = max_seq_len
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size, padding = kernel_size // 2),
                nn.GELU()
            ))
        self.to_logits = nn.Conv1d(dim, num_tokens, 1)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.to_logits(x)
    
class MotifEmbedder(nn.Module):
    """Conv1D block that map motif_dim to hidden_dim"""
    def __init__(self, motif_dim, hidden_dim, kernel_size = 1):
        super().__init__()
        self.conv = nn.Conv1d(motif_dim, hidden_dim, kernel_size)
    def forward(self, x):
        return self.conv(x)



class ConvPool(nn.Module):
    """
    Receive a tensor of shape (batch, length, dimension) and split along length
    dimension based on a celltype_peak tensor of shape (batch, n_peak, 2) where
    the second dimension is the start and end of the peak. The length dimension 
    can be decomposed into a sum of the peak lengths with each peak padded left 
    and right with 50bp and directly concatenated. Thus the boundary for the 
    splitting can be calculated by cumsum of the padded peak lengths. The output
    is a tensor of shape (batch, n_peak, dimension). 
    """
    def __init__(self, motif_dim, hidden_dim, pool_method='mean'):
        super().__init__()
        self.pool_method = pool_method
        self.motif_proj = nn.Conv1d(motif_dim, hidden_dim, 1, bias=False)
        self.dila_conv_tower = DilatedConv1dBlock(hidden_dim, depth = 7)
        self.aprofile_header = nn.Conv1d(hidden_dim, 1, 20, padding='same')
        self.atpm_header = nn.Conv1d(hidden_dim, 1, 1, padding='same')
        
    def forward(self, x, peak_split, n_peaks, max_n_peaks):
        """
        x: (batch, length, dimension)
        chunk_size: the size of each chunk to pool        
        n_peaks: the number of peaks for each sample
        max_n_peaks: the maximum number of peaks in the batch
        pool_method: the method to pool the tensor
        """

        # linear prob motif_dim to 512
        x = self.motif_proj(x.transpose(1,2)).transpose(1,2)
        batch, length, motif_dim = x.shape
        # split the tensor
        peak_list = torch.split(x.reshape(-1,motif_dim), peak_split, dim=0)
        # each element is L, D, pool the tensor

        peak_atpm_list = []
        peak_profiles = []
        for peak in peak_list:
            peak_profile = self.dila_conv_tower(peak)
            peak_profile = self.aprofile_header(peak_profile)
            peak_atpm = self.pool(peak_profile, self.pool_method)
            peak_atpm = self.atpm_header(peak_atpm)
            peak_atpm_list.append(peak_atpm)
            peak_profiles.append(peak_profile)
        peak_atpm_list = torch.vstack(peak_atpm_list)
        peak_profiles = torch.vstack(peak_profiles)
        # remove the padded part
        pool_idx = torch.cumsum(n_peaks+1,0)
        pool_left_pad = torch.tensor(0).unsqueeze(0).to(pool_idx.device)
        pool_start = torch.cat([pool_left_pad, pool_idx[:-1]])
        pool_end = pool_idx-1
        pool_list = [peak_list[pool_start[i]:pool_end[i]] for i in range(len(pool_start))]
        # pad the element in pool_list if the number of peaks is not the same
        peak_atpms = torch.stack([torch.cat([pool_list[i], torch.zeros(max_n_peaks-n_peaks[i], motif_dim).to(pool_list[i].device)]) for i in range(len(pool_list))])

        return peak_atpms, peak_profiles

class ATACSplitPool(nn.Module):
    """
    Receive a tensor of shape (batch, length, dimension) and split along length
    dimension based on a celltype_peak tensor of shape (batch, n_peak, 2) where
    the second dimension is the start and end of the peak. The length dimension 
    can be decomposed into a sum of the peak lengths with each peak padded left 
    and right with 50bp and directly concatenated. Thus the boundary for the 
    splitting can be calculated by cumsum of the padded peak lengths. The output
    is a tensor of shape (batch, n_peak, dimension). 
    """
    def __init__(self, pool_method='mean', atac_kernel_num=16, motif_dim=639, joint_kernel_num=16, atac_kernel_size=3, joint_kernel_size=3, final_bn=False, binary_atac=False):
        super().__init__()
        self.pool_method = pool_method
        self.atac_conv = nn.Conv1d(1, atac_kernel_num, atac_kernel_size, padding="same", bias=False)
        self.atac_bn = nn.BatchNorm1d(atac_kernel_num, affine=False)
        self.joint_conv = nn.Conv1d(motif_dim + atac_kernel_num, joint_kernel_num, joint_kernel_size, padding="same", bias=False)
        self.joint_bn = nn.BatchNorm1d(joint_kernel_num, affine=False)
        self.patch_pool = nn.MaxPool1d(25, stride=25)
        if final_bn:
            self.final_bn = nn.BatchNorm1d(motif_dim + joint_kernel_num, affine=False)
        self.binary_atac = binary_atac


    def forward(self, x, atac, peak_split, n_peaks, max_n_peaks):
        # normalize atac to [0,1], keeps mostly shape information
        # atac = atac / (atac.max(1, keepdim=True)[0]+1e-5)
        if self.binary_atac:
            atac = (atac > 0).float()
        else:
            atac = torch.log10(atac+1)
        # split pool motif signal to region level
        x_region = self.forward_x(x, peak_split, n_peaks, max_n_peaks)
        # jointly convolve atac and motif signal at 50bp bin level
        joint_region = self.forward_joint(x, atac, peak_split, n_peaks, max_n_peaks)
        # log transform to make the signal < 10
        joint_region = torch.log2(joint_region+1)
        # concatenate motif representation with joint representation
        # shape (batch, n_peak, motif_dim + joint_kernel_num)
        x = torch.cat([x_region, joint_region], dim=2).contiguous()
        # batch norm
        if hasattr(self, 'final_bn'):
            x = self.final_bn(x.transpose(1,2)).transpose(1,2)
        return x

    def forward_joint(self, x, atac, peak_split, n_peaks, max_n_peaks, patch_size=25):
        """
        x: (batch, length, dimension)
        atac: (batch, length, 1)
        patch_size: the size of each chunk to pool
        n_peaks: the number of peaks for each sample
        max_n_peaks: the maximum number of peaks in the batch
        pool_method: the method to pool the tensor
        """
        x = x.transpose(1,2).contiguous()
        atac = atac.unsqueeze(1).contiguous()
        x_pooled = self.patch_pool(x)
        atac_pooled = self.patch_pool(atac)
        # shrink peak_split according to patch_size
        patch_peak_split = [i//patch_size for i in peak_split]
        # remove 0s in peak_split
        # convolve atac
        atac_pooled = self.atac_conv(atac_pooled)
        atac_pooled = self.atac_bn(atac_pooled)
        atac_pooled = F.relu(atac_pooled)

        # atac = torch.cat([atac, atac_pooled], dim=1)
        # concatenate atac and x
        x_pooled = torch.cat([x_pooled, atac_pooled], dim=1).contiguous()
        # convolve x_pooled
        x_pooled = self.joint_conv(x_pooled) 
        x_pooled = self.joint_bn(x_pooled) # (B, D, L//50)
        # relu
        x_pooled = F.relu(x_pooled).transpose(1,2) # (B, L//50, D)
        batch, length, embed_dim = x_pooled.shape
        # further mean pool based on peak_split
        chunk_list = torch.split(x_pooled.reshape(-1,embed_dim), patch_peak_split, dim=0)
        # each element is L, D, pool the tensor
        chunk_list = torch.vstack([pool(chunk, self.pool_method) for chunk in chunk_list])
        # remove the padded part
        pool_idx = torch.cumsum(n_peaks+1,0)
        pool_left_pad = torch.tensor(0).unsqueeze(0).to(pool_idx.device)
        pool_start = torch.cat([pool_left_pad, pool_idx[:-1]])
        pool_end = pool_idx-1
        pool_list = [chunk_list[pool_start[i]:pool_end[i]] for i in range(len(pool_start))]
        # pad the element in pool_list if the number of peaks is not the same
        x_pooled = torch.stack([torch.cat([pool_list[i], torch.zeros(max_n_peaks-n_peaks[i], embed_dim).to(pool_list[i].device)]) for i in range(len(pool_list))]) # (B, R, D)
        return x_pooled

    def forward_x(self, x, peak_split, n_peaks, max_n_peaks):
        """
        x: (batch, length, dimension)
        chunk_size: the size of each chunk to pool        
        n_peaks: the number of peaks for each sample
        max_n_peaks: the maximum number of peaks in the batch
        pool_method: the method to pool the tensor
        """
        batch, length, embed_dim = x.shape
        chunk_list = torch.split(x.reshape(-1,embed_dim), peak_split, dim=0)
        # each element is L, D, pool the tensor
        chunk_list = torch.vstack([pool(chunk, self.pool_method) for chunk in chunk_list])
        # remove the padded part
        pool_idx = torch.cumsum(n_peaks+1,0)
        pool_left_pad = torch.tensor(0).unsqueeze(0).to(pool_idx.device)
        pool_start = torch.cat([pool_left_pad, pool_idx[:-1]])
        pool_end = pool_idx-1
        pool_list = [chunk_list[pool_start[i]:pool_end[i]] for i in range(len(pool_start))]
        # pad the element in pool_list if the number of peaks is not the same
        x = torch.stack([torch.cat([pool_list[i], torch.zeros(max_n_peaks-n_peaks[i], embed_dim).to(pool_list[i].device)]) for i in range(len(pool_list))]) # (B, R, D)
        return x


class ATACSplitPoolMaxNorm(nn.Module):
    """
    Receive a tensor of shape (batch, length, dimension) and split along length
    dimension based on a celltype_peak tensor of shape (batch, n_peak, 2) where
    the second dimension is the start and end of the peak. The length dimension 
    can be decomposed into a sum of the peak lengths with each peak padded left 
    and right with 50bp and directly concatenated. Thus the boundary for the 
    splitting can be calculated by cumsum of the padded peak lengths. The output
    is a tensor of shape (batch, n_peak, dimension). 
    """
    def __init__(self, pool_method='mean', atac_kernel_num=16, motif_dim=639, joint_kernel_num=16, atac_kernel_size=3, joint_kernel_size=3, final_bn=False, atac_input_norm=True):
        super().__init__()
        self.pool_method = pool_method
        self.atac_input_norm = atac_input_norm
        self.motif_dim = motif_dim
        self.joint_dim = joint_kernel_num
        self.atac_conv = nn.Conv1d(1, atac_kernel_num, atac_kernel_size, padding="same", bias=False)
        self.atac_bn = nn.BatchNorm1d(atac_kernel_num, affine=False)
        self.joint_conv = nn.Conv1d(motif_dim + atac_kernel_num, joint_kernel_num, joint_kernel_size, padding="same", bias=False)
        # a running max of motif signal, support DDP
        self.register_buffer('running_max', torch.ones(motif_dim))
        self.register_buffer('running_max_joint', torch.ones(joint_kernel_num))
        self.joint_bn = nn.BatchNorm1d(joint_kernel_num, affine=False)
        self.patch_pool = nn.MaxPool1d(25, stride=25)
        if final_bn:
            self.final_bn = nn.BatchNorm1d(motif_dim + joint_kernel_num, affine=False)


    def forward(self, x, x_region, atac, peak_split, n_peaks, max_n_peaks):
        # normalize atac to [0,1], keeps mostly shape information
        # atac = atac / (atac.max(1, keepdim=True)[0]+1e-5)
        # atac = torch.log2(atac+1)
        if self.atac_input_norm:
            atac = atac / (atac.max(1, keepdim=True)[0]+1e-5)
        # split pool motif signal to region level
        # jointly convolve atac and motif signal at 50bp bin level
        with torch.no_grad():
            self.running_max = torch.max(self.running_max, torch.max(x_region.view(-1, self.motif_dim), dim=0).values)
        x_region = x_region / (self.running_max.unsqueeze(0).unsqueeze(0)+1e-5)
        joint_region = self.forward_joint(x, atac, peak_split, n_peaks, max_n_peaks)
        with torch.no_grad():
            self.running_max_joint = torch.max(self.running_max_joint, torch.max(joint_region.view(-1, self.joint_dim), dim=0).values)
        joint_region = joint_region / (self.running_max_joint.unsqueeze(0).unsqueeze(0)+1e-5)

        # log transform to make the signal < 10
        # joint_region = torch.log2(joint_region+1)
        # concatenate motif representation with joint representation
        # shape (batch, n_peak, motif_dim + joint_kernel_num)
        x = torch.cat([x_region, joint_region], dim=2).contiguous()
        # batch norm
        if hasattr(self, 'final_bn'):
            x = self.final_bn(x.transpose(1,2)).transpose(1,2)
        return x

    def forward_joint(self, x, atac, peak_split, n_peaks, max_n_peaks, patch_size=25):
        """
        x: (batch, length, dimension)
        atac: (batch, length, 1)
        patch_size: the size of each chunk to pool
        n_peaks: the number of peaks for each sample
        max_n_peaks: the maximum number of peaks in the batch
        pool_method: the method to pool the tensor
        """
        x = x.transpose(1,2).contiguous()
        atac = atac.unsqueeze(1).contiguous()
        x_pooled = self.patch_pool(x)
        atac_pooled = self.patch_pool(atac)
        # shrink peak_split according to patch_size
        patch_peak_split = [i//patch_size for i in peak_split]
        # remove 0s in peak_split
        # convolve atac
        atac_pooled = self.atac_conv(atac_pooled)
        atac_pooled = self.atac_bn(atac_pooled)
        atac_pooled = F.relu(atac_pooled)

        # atac = torch.cat([atac, atac_pooled], dim=1)
        # concatenate atac and x
        x_pooled = torch.cat([x_pooled, atac_pooled], dim=1).contiguous()
        # convolve x_pooled
        x_pooled = self.joint_conv(x_pooled) 
        x_pooled = self.joint_bn(x_pooled) # (B, D, L//50)
        # relu
        x_pooled = F.relu(x_pooled).transpose(1,2) # (B, L//50, D)
        batch, length, embed_dim = x_pooled.shape
        # further mean pool based on peak_split
        chunk_list = torch.split(x_pooled.reshape(-1,embed_dim), patch_peak_split, dim=0)
        # each element is L, D, pool the tensor
        chunk_list = torch.vstack([pool(chunk, self.pool_method) for chunk in chunk_list])
        # remove the padded part
        pool_idx = torch.cumsum(n_peaks+1,0)
        pool_left_pad = torch.tensor(0).unsqueeze(0).to(pool_idx.device)
        pool_start = torch.cat([pool_left_pad, pool_idx[:-1]])
        pool_end = pool_idx-1
        pool_list = [chunk_list[pool_start[i]:pool_end[i]] for i in range(len(pool_start))]
        # pad the element in pool_list if the number of peaks is not the same
        x_pooled = torch.stack([torch.cat([pool_list[i], torch.zeros(max_n_peaks-n_peaks[i], embed_dim).to(pool_list[i].device)]) for i in range(len(pool_list))]) # (B, R, D)
        return x_pooled

    def forward_x(self, x, peak_split, n_peaks, max_n_peaks):
        """
        x: (batch, length, dimension)
        chunk_size: the size of each chunk to pool        
        n_peaks: the number of peaks for each sample
        max_n_peaks: the maximum number of peaks in the batch
        pool_method: the method to pool the tensor
        """
        batch, length, embed_dim = x.shape
        chunk_list = torch.split(x.reshape(-1,embed_dim), peak_split, dim=0)
        # each element is L, D, pool the tensor
        chunk_list = torch.vstack([pool(chunk, self.pool_method) for chunk in chunk_list])
        # remove the padded part
        pool_idx = torch.cumsum(n_peaks+1,0)
        pool_left_pad = torch.tensor(0).unsqueeze(0).to(pool_idx.device)
        pool_start = torch.cat([pool_left_pad, pool_idx[:-1]])
        pool_end = pool_idx-1
        pool_list = [chunk_list[pool_start[i]:pool_end[i]] for i in range(len(pool_start))]
        # pad the element in pool_list if the number of peaks is not the same
        x = torch.stack([torch.cat([pool_list[i], torch.zeros(max_n_peaks-n_peaks[i], embed_dim).to(pool_list[i].device)]) for i in range(len(pool_list))]) # (B, R, D)
        return x



class AttentionPooling(nn.Module):
    # Copied from https://github.com/deep-floyd/IF/blob/2f91391f27dd3c468bf174be5805b4cc92980c0b/deepfloyd_if/model/nn.py#L54

    def __init__(self, num_heads, embed_dim, dtype=None):
        super().__init__()
        self.dtype = dtype
        self.positional_embedding = nn.Parameter(torch.randn(1, embed_dim) / embed_dim**0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim, dtype=self.dtype)
        self.q_proj = nn.Linear(embed_dim, embed_dim, dtype=self.dtype)
        self.v_proj = nn.Linear(embed_dim, embed_dim, dtype=self.dtype)
        self.num_heads = num_heads
        self.dim_per_head = embed_dim // self.num_heads

    def forward(self, x):
        bs, length, width = x.size()

        def shape(x):
            # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
            x = x.view(bs, -1, self.num_heads, self.dim_per_head)
            # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
            x = x.transpose(1, 2)
            # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
            x = x.reshape(bs * self.num_heads, -1, self.dim_per_head)
            # (bs*n_heads, length, dim_per_head) --> (bs*n_heads, dim_per_head, length)
            x = x.transpose(1, 2)
            return x

        class_token = x.mean(dim=1, keepdim=True) + self.positional_embedding.to(x.dtype)
        x = torch.cat([class_token, x], dim=1)  # (bs, length+1, width)

        # (bs*n_heads, class_token_length, dim_per_head)
        q = shape(self.q_proj(class_token))
        # (bs*n_heads, length+class_token_length, dim_per_head)
        k = shape(self.k_proj(x))
        v = shape(self.v_proj(x))

        # (bs*n_heads, class_token_length, length+class_token_length):
        scale = 1 / math.sqrt(math.sqrt(self.dim_per_head))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)

        # (bs*n_heads, dim_per_head, class_token_length)
        a = torch.einsum("bts,bcs->bct", weight, v)

        # (bs, length+1, width)
        a = a.reshape(bs, -1, 1).transpose(1, 2)

        return a[:, 0, :]  # cls_token