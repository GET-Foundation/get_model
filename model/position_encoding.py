import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pyranges import PyRanges as pr


class CTCFPositionalEncoding(nn.Module):
    def __init__(self, hidden, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        # ctcf_pos: Tensor, shape [batch_size, 5, seq_len], replace the position variable
        self.dropout = nn.Dropout(p=dropout)
        div_term = torch.exp(torch.arange(0, (hidden), 2) * (-np.log(300.0) / (hidden)))
        self.register_buffer("div_term", div_term)

    def forward(self, x, ctcf_pos):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        pos_embeds = []
        # (B, 200, 5) ctcf_pos
        for i in range(ctcf_pos.shape[2]):
            pos_embed = torch.zeros(x.size(0), x.size(1), self.hidden).to(x.device)
            pos_embed[:, :, 0::2] = torch.sin(
                ctcf_pos[:, :, i].unsqueeze(2) * self.div_term
            )
            pos_embed[:, :, 1::2] = torch.cos(
                ctcf_pos[:, :, i].unsqueeze(2) * self.div_term
            )
            pos_embeds.append(pos_embed)
        # mean of N multi-level ctcf_pos
        pos_embed = torch.mean(torch.stack(pos_embeds), dim=0)
        x = x + pos_embed
        return self.dropout(x)


class AbsolutePositionalEncoding(nn.Module):
    """Absolute positional encoding, as described in "Attention Is All You Need"."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 200):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(300.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)
