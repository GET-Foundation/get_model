import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn

def pool(x, method='mean'):
    """
    x: (L,D)
    """
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
    def __init__(self, pool_method='sum'):
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
        pool_start = torch.cat([torch.tensor(0).unsqueeze(0), pool_idx[:-1]])
        pool_end = pool_idx-1
        pool_list = [chunk_list[pool_start[i]:pool_end[i]] for i in range(len(pool_start))]
        # pad the element in pool_list if the number of peaks is not the same
        x = torch.stack([torch.cat([pool_list[i], torch.zeros(max_n_peaks-n_peaks[i], embed_dim)]) for i in range(len(pool_list))])

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