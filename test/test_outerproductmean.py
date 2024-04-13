# %%
from tqdm import tqdm
import torch
import torch.nn as nn
from typing import Optional


class OuterProductMean(nn.Module):
    """
    Implements a simplified version of the OuterProductMean.
    """

    def __init__(self, c_m, c_z, c_hidden, eps=1e-3):
        """
        Args:
            c_m: region embedding channel dimension
            c_z: Pair embedding channel dimension
            c_hidden: Hidden channel dimension
        """
        super(OuterProductMean, self).__init__()
        self.eps = eps
        self.layer_norm = nn.LayerNorm(c_m)
        self.linear_1 = nn.Linear(c_m, c_hidden)
        self.linear_2 = nn.Linear(c_m, c_hidden)
        self.linear_out = nn.Linear(c_hidden ** 2, c_z)

    def forward(self, m: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            m: [*, N_region, C_m] region embedding
            mask: [*, N_region] MSA mask, if None, create a mask of ones
        Returns:
            [*, N_region, N_region, C_z] pair embedding update
        """
        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        ln = self.layer_norm(m)
        mask = mask.unsqueeze(-1)

        a = self.linear_1(ln) * mask
        b = self.linear_2(ln) * mask

        # Calculate the outer product mean -> [batch, N, N, C, C]
        outer = torch.einsum("...bc,...de->...bdce", a, b)
        outer = outer.reshape(outer.shape[:-2] + (-1,))
        outer = self.linear_out(outer)
        norm = torch.einsum("...b,...d->...bd", mask, mask) + self.eps
        outer = outer / norm

        return outer


# %%
input = torch.randn(20, 900, 768).to("cuda")
# %%
model = OuterProductMean(768, 128, 32).to("cuda")

# %%
for i in tqdm(range(10000)):
    output = model(input)
    # output.backward(torch.randn_like(output))
# %%
