import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn

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
    def __init__(self, pool_method='mean', atac_kernel_num=16, motif_dim=639, joint_kernel_num=16, atac_kernel_size=3, joint_kernel_size=3, final_bn=False):
        super().__init__()
        self.pool_method = pool_method
        self.atac_conv = nn.Conv1d(1, atac_kernel_num, atac_kernel_size, padding="same", bias=False)
        self.atac_bn = nn.BatchNorm1d(atac_kernel_num, affine=False)
        self.joint_conv = nn.Conv1d(motif_dim + atac_kernel_num, joint_kernel_num, joint_kernel_size, padding="same", bias=False)
        self.joint_bn = nn.BatchNorm1d(joint_kernel_num, affine=False)
        self.patch_pool = nn.MaxPool1d(25, stride=25)
        if final_bn:
            self.final_bn = nn.BatchNorm1d(motif_dim + joint_kernel_num, affine=False)


    def forward(self, x, atac, peak_split, n_peaks, max_n_peaks):
        # normalize atac to [0,1], keeps mostly shape information
        # atac = atac / (atac.max(1, keepdim=True)[0]+1e-5)
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


class AttentionPool(nn.Module):
    def __init__(self, dim, pool_size = 2):
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn = Rearrange('b d (n p) -> b d n p', p = pool_size)

        self.to_attn_logits = nn.Conv2d(dim, dim, 1, bias = False)

        nn.init.dirac_(self.to_attn_logits.weight)

        with torch.no_grad():
            self.to_attn_logits.weight.mul_(2)

    def forward(self, x):
        b, _, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0

        if needs_padding:
            x = F.pad(x, (0, remainder), value = 0)
            mask = torch.zeros((b, 1, n), dtype = torch.bool, device = x.device)
            mask = F.pad(mask, (0, remainder), value = True)

        x = self.pool_fn(x)
        logits = self.to_attn_logits(x)

        if needs_padding:
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(self.pool_fn(mask), mask_value)

        attn = logits.softmax(dim = -1)

        return (x * attn).sum(dim = -1)
