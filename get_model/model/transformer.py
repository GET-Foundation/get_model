from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from timm.models.layers import DropPath
try:
    from flash_attn import flash_attn_qkvpacked_func
except ImportError:
    flash_attn_qkvpacked_func = None


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_head_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attention_mask=None, attention_bias=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        # get dtype of attn
        attn_dtype = attn.dtype
        # determine the mask value based on the dtype of attention
        if attn_dtype == torch.float16:
            attn_mask_value = -65504
        elif attn_dtype == torch.float32:
            attn_mask_value = -65504
        elif attn_dtype == torch.bfloat16:
            attn_mask_value = -65504

        if attention_mask is not None:
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(
                    0).unsqueeze(0).unsqueeze(0)
            elif attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            assert (
                attention_mask.shape[-1] == N
            ), "The last dimension of attention_mask should be equal to N. Currently the shape is {}".format(
                attention_mask.shape
            )

            attn = attn.masked_fill(attention_mask, attn_mask_value)
        if attention_bias is not None:
            if attention_bias.dim() == 1:
                attention_bias = attention_bias.unsqueeze(
                    0).unsqueeze(0).unsqueeze(0)
            elif attention_bias.dim() == 2:
                attention_bias = attention_bias.unsqueeze(1).unsqueeze(1)
            elif attention_bias.dim() == 3:
                attention_bias = attention_bias.unsqueeze(1)

            attn = attn + attention_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Attention_Flash(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_head_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attention_mask=None):
        # TODO: add attention mask support
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1)
        x = flash_attn_qkvpacked_func(
            qkv, softmax_scale=self.scale).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, None


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.1,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.1,
        attn_drop=0.1,
        drop_path=0.1,
        init_values=0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_head_dim=None,
        flash_attn=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if flash_attn:
            self.attn = Attention_Flash(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                attn_head_dim=attn_head_dim,
            )
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                attn_head_dim=attn_head_dim,
            )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if init_values > 0:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, attention_mask=None, attention_bias=None):
        if self.gamma_1 is None:
            x_attn, attn = self.attn(self.norm1(
                x), attention_mask=attention_mask, attention_bias=attention_bias)
            x = x + self.drop_path(x_attn)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x_attn, attn = self.attn(self.norm1(
                x), attention_mask=attention_mask, attention_bias=attention_bias)
            x = x + self.drop_path(self.gamma_1 * x_attn)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x, attn


class GETTransformer(nn.Module):
    """A transformer module for GET model."""

    def __init__(
        self,
        embed_dim,
        num_heads=8,
        num_layers=8,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path_rate=0.1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0,
        use_mean_pooling=False,
        flash_attn=False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, num_layers)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                    flash_attn=flash_attn,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None

    def forward(self, x, mask=None, return_attns=False, bias=None):
        attn_output = [] if return_attns else None
        for blk in self.blocks:
            x, attn = blk(x, mask, bias)
            if return_attns:
                attn_output.append(attn)
        x = self.norm(x)
        if self.fc_norm is not None:
            x = self.fc_norm(x.mean(1))
        return x, attn_output


class OuterProductMean(nn.Module):
    """
    Implements a simplified version of the OuterProductMean.
    """

    def __init__(self, c_m, c_z, c_hidden, eps=1e-3):
        """
        Args:
            c_m: MSA embedding channel dimension
            c_z: Pair embedding channel dimension
            c_hidden: Hidden channel dimension
        """
        super(OuterProductMean, self).__init__()

        self.eps = eps
        self.layer_norm = nn.LayerNorm(c_m)
        self.linear_1 = nn.Linear(c_m, c_hidden)
        self.linear_2 = nn.Linear(c_m, c_hidden)
        self.linear_out = nn.Linear(c_hidden ** 2, c_z, init="final")

    def forward(self, m: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            m: [*, N_seq, N_res, C_m] MSA embedding
            mask: [*, N_seq, N_res] MSA mask, if None, create a mask of ones
        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        ln = self.layer_norm(m)
        mask = mask.unsqueeze(-1)
        a = self.linear_1(ln) * mask
        b = self.linear_2(ln) * mask
        a = a.transpose(-2, -3)
        b = b.transpose(-2, -3)

        # Calculate the outer product mean
        outer = torch.einsum("...bac,...dae->...bdce", a, b)
        outer = outer.reshape(outer.shape[:-2] + (-1,))
        outer = self.linear_out(outer)

        norm = torch.einsum("...abc,...adc->...bdc", mask, mask) + self.eps
        outer = outer / norm

        return outer
