from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from timm.models.layers import DropPath, trunc_normal_

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
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
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
                attention_bias = attention_bias.unsqueeze(0).unsqueeze(0).unsqueeze(0)
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
        x = flash_attn_qkvpacked_func(qkv, softmax_scale=self.scale).reshape(B, N, C)
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
        drop=0.0,
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
        # x = self.drop1(x)
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
        drop=0,
        attn_drop=0,
        drop_path=0.1,
        init_values=0.001,
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
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
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
            x_attn, attn = self.attn(
                self.norm1(x),
                attention_mask=attention_mask,
                attention_bias=attention_bias,
            )
            x = x + self.drop_path(x_attn)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x_attn, attn = self.attn(
                self.norm1(x),
                attention_mask=attention_mask,
                attention_bias=attention_bias,
            )
            x = x + self.drop_path(self.gamma_1 * x_attn)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x, attn


try:
    from axial_attention import AxialAttention

    class PairedBlock(nn.Module):
        """A block that will perform the axial attention and mlp on the paired embedding."""

        def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            drop=0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
        ):
            super().__init__()
            self.norm1 = norm_layer(dim)
            self.axial_attn = AxialAttention(
                dim,
                num_dimensions=2,
                heads=num_heads,
            )
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )

        def forward(self, x, attention_mask=None, attention_bias=None):
            x_attn = self.axial_attn(self.norm1(x))
            x = x + x_attn
            x = x + self.mlp(self.norm2(x))
            return x

except ImportError:
    PairedBlock = None


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
        drop_rate=0,
        attn_drop_rate=0,
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

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

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


class GETTransformerWithContactMap(GETTransformer):
    """A transformer module for GET model that takes a distance map as an additional input and it will fuse every layer of GET base model pairwise embedding to the distance map."""

    def __init__(
        self,
        embed_dim,
        num_heads=8,
        num_layers=8,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0,
        attn_drop_rate=0,
        drop_path_rate=0.1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0,
        use_mean_pooling=False,
        flash_attn=False,
        *args,
        **kwargs,
    ):
        super().__init__(
            embed_dim,
            num_heads,
            num_layers,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            norm_layer,
            init_values,
            use_mean_pooling,
            flash_attn,
            *args,
            **kwargs,
        )

    def forward(self, x, distance_map, mask=None, return_attns=False, bias=None):
        attn_output = [] if return_attns else None
        distance_map = distance_map.squeeze(1).unsqueeze(3)
        for blk in self.blocks:
            x, attn = blk(x, mask, bias)
            if return_attns:
                attn_output.append(attn)

            # Perform outer sum of x
            outer_sum = x[:, 1:].unsqueeze(2).detach() + x[:, :-1].unsqueeze(1).detach()
            # Sum the outer_sum with distance_map with stop_grad to prevent backpropagation from distance map to embedding
            distance_map = distance_map + outer_sum / len(self.blocks)

        x = self.norm(x)
        if self.fc_norm is not None:
            x = self.fc_norm(x.mean(1))

        return x, distance_map, attn_output


class GETTransformerWithContactMapOE(GETTransformer):
    """A transformer module for GET model that takes a distance map as an additional input and it will fuse every layer of GET base model pairwise embedding to the distance map."""

    def __init__(
        self,
        embed_dim,
        num_heads=8,
        num_layers=8,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0,
        attn_drop_rate=0,
        drop_path_rate=0.1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0,
        use_mean_pooling=False,
        flash_attn=False,
        *args,
        **kwargs,
    ):
        super().__init__(
            embed_dim,
            num_heads,
            num_layers,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            norm_layer,
            init_values,
            use_mean_pooling,
            flash_attn,
            *args,
            **kwargs,
        )

    def forward(self, x, distance_map, mask=None, return_attns=False, bias=None):
        attn_output = [] if return_attns else None
        distance_map = distance_map.squeeze(1)
        # concat 1 to first row and column of distance map (B, R, R, 1) to (B, R+1, R+1, 1)
        # bias = F.pad(distance_map, (0, 1, 0, 1), "constant", 0)
        # bias = bias.unsqueeze(1)

        for blk in self.blocks:
            x, attn = blk(x, mask)
            if return_attns:
                attn_output.append(attn)

        x = self.norm(x)
        # Perform outer sum of x
        if self.fc_norm is not None:
            x = self.fc_norm(x.mean(1))

        outer_sum = x[:, 1:].unsqueeze(2) + x[:, :-1].unsqueeze(1)
        # concat distance map to outer sum
        outer_sum = torch.cat([distance_map.unsqueeze(3), outer_sum], dim=3)

        return x, outer_sum, attn_output


class GETTransformerWithContactMapAxial(GETTransformer):
    """A transformer module for GET model that takes a distance map as an additional input and it will fuse every layer of GET base model pairwise embedding to the distance map."""

    def __init__(
        self,
        embed_dim,
        num_heads=8,
        num_layers=8,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0,
        attn_drop_rate=0,
        drop_path_rate=0.1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0,
        use_mean_pooling=False,
        flash_attn=False,
        *args,
        **kwargs,
    ):
        super().__init__(
            embed_dim,
            num_heads,
            num_layers,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            norm_layer,
            init_values,
            use_mean_pooling,
            flash_attn,
            *args,
            **kwargs,
        )
        self.paired_blocks = nn.ModuleList(
            [
                PairedBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    act_layer=nn.GELU,
                    norm_layer=norm_layer,
                )
                for i in range(num_layers // 3)
            ]
        )

    def forward(self, x, distance_map, mask=None, return_attns=False, bias=None):
        attn_output = [] if return_attns else None
        distance_map = distance_map.squeeze(1).unsqueeze(3)
        for i, blk in enumerate(self.blocks):
            x, attn = blk(x, mask, bias)
            if return_attns:
                attn_output.append(attn)

            # Perform outer sum of x
            if i % 2 == 0 and i < len(self.paired_blocks):
                outer_sum = (
                    x[:, 1:].unsqueeze(2).detach() + x[:, :-1].unsqueeze(1).detach()
                )
                # Sum the outer_sum with distance_map with stop_grad to prevent backpropagation from distance map to embedding
                distance_map = distance_map + outer_sum
                # add axial attention
                distance_map = self.paired_blocks[i // 2](distance_map)

        x = self.norm(x)
        distance_map = F.gelu(distance_map)
        if self.fc_norm is not None:
            x = self.fc_norm(x.mean(1))

        return x, distance_map, attn_output
