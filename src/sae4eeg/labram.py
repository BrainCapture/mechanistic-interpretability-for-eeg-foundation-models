"""
labram.py — Vendored LaBraM (Jiang et al., ICLR 2024) architecture.

Lightly adapted from the official implementation at
https://github.com/935963004/LaBraM/blob/main/modeling_finetune.py
(file ``modeling_finetune.py``). The only changes from upstream:

* ``timm`` is dropped — ``drop_path`` is inlined and ``trunc_normal_`` is
  taken from ``torch.nn.init``.
* The ``@register_model`` decorator from timm is dropped.
* ``PatchEmbed`` (used only for the neural-decoder path during pretraining)
  is dropped — finetuning always uses ``TemporalConv``.
* Comments and dead branches are pruned.

The core ``NeuralTransformer`` and its ``Block`` / ``Attention`` /
``TemporalConv`` submodules are preserved verbatim so the upstream
state_dict format can be loaded directly. Use ``labram_base_patch200_200``
to construct LaBraM-Base.

Used by ``LaBraMBackend`` in ``encoders.py``.
"""
from __future__ import annotations

import math
from functools import partial
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Inlined timm.drop_path — stochastic depth per sample."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


def trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    return nn.init.trunc_normal_(tensor, std=std)


class DropPath(nn.Module):
    def __init__(self, drop_prob: Optional[float] = None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob or 0.0, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None, act_layer=nn.GELU, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 qk_norm=None, qk_scale: Optional[float] = None,
                 attn_drop: float = 0.0, proj_drop: float = 0.0, attn_head_dim: Optional[int] = None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if qk_norm is not None:
            self.q_norm = qk_norm(head_dim)
            self.k_norm = qk_norm(head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, return_attention: bool = False, return_qkv: bool = False):
        B, N, _ = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias)
            )
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.q_norm is not None:
            q = self.q_norm(q).type_as(v)
        if self.k_norm is not None:
            k = self.k_norm(k).type_as(v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        attn = self.attn_drop(attn)

        if return_attention:
            return attn

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_qkv:
            return x, qkv
        return x


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = False, qk_norm=None, qk_scale: Optional[float] = None,
                 drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.0,
                 init_values: Optional[float] = None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim: Optional[int] = None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class TemporalConv(nn.Module):
    """EEG → patch embedding via three 1×k temporal convs with GroupNorm + GELU."""

    def __init__(self, in_chans: int = 1, out_chans: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, A, T) where N=n_electrodes, A=n_patches, T=patch_size
        x = rearrange(x, "B N A T -> B (N A) T")
        x = x.unsqueeze(1)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        x = rearrange(x, "B C NA T -> B NA (T C)")
        return x


class NeuralTransformer(nn.Module):
    def __init__(
        self,
        EEG_size: int = 1600,
        patch_size: int = 200,
        in_chans: int = 1,
        out_chans: int = 8,
        num_classes: int = 0,
        embed_dim: int = 200,
        depth: int = 12,
        num_heads: int = 10,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm=None,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer=nn.LayerNorm,
        init_values: Optional[float] = None,
        use_abs_pos_emb: bool = True,
        init_scale: float = 0.001,
        max_channels: int = 128,
        max_time_patches: int = 16,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = TemporalConv(in_chans=1, out_chans=out_chans)
        self.time_window = EEG_size // patch_size
        self.patch_size = patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, max_channels + 1, embed_dim), requires_grad=True)
        else:
            self.pos_embed = None
        self.time_embed = nn.Parameter(torch.zeros(1, max_time_patches, embed_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, init_values=init_values,
            )
            for i in range(depth)
        ])
        self.norm = nn.Identity()
        self.fc_norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        if self.time_embed is not None:
            trunc_normal_(self.time_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=0.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self) -> int:
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "time_embed"}

    def forward_features(
        self,
        x: torch.Tensor,
        input_chans: Optional[List[int]] = None,
        return_patch_tokens: bool = False,
        return_all_tokens: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, N, A, T) where N=n_electrodes, A=n_patches, T=patch_size.
            Also accepts (B, C, T) with T = A * patch_size; reshapes internally.
        input_chans : list[int] or None
            Indices into ``pos_embed[:, 1:]`` for the C channels in ``x``.
            ``None`` → use the first C positions.
        """
        if x.dim() == 3:
            B, C, T = x.shape
            assert T % self.patch_size == 0, f"T={T} not divisible by patch_size={self.patch_size}"
            A = T // self.patch_size
            x = x.reshape(B, C, A, self.patch_size)
        batch_size, n, a, t = x.shape
        input_time_window = a if t == self.patch_size else t

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed is not None:
            if input_chans is not None:
                chan_idx = torch.tensor(input_chans, device=x.device, dtype=torch.long)
                pos_used = self.pos_embed[:, chan_idx + 1, :]      # exclude cls index
                pos_cls = self.pos_embed[:, 0:1, :]
            else:
                pos_used = self.pos_embed[:, 1:1 + n, :]
                pos_cls = self.pos_embed[:, 0:1, :]
            pos_used = pos_used.unsqueeze(2).expand(batch_size, -1, input_time_window, -1).flatten(1, 2)
            pos_used = torch.cat((pos_cls.expand(batch_size, -1, -1), pos_used), dim=1)
            x = x + pos_used
        if self.time_embed is not None:
            nc = n if t == self.patch_size else a
            time_embed = self.time_embed[:, :input_time_window, :].unsqueeze(1).expand(batch_size, nc, -1, -1).flatten(1, 2)
            x[:, 1:, :] = x[:, 1:, :] + time_embed

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        if return_all_tokens:
            return self.fc_norm(x)
        t_tokens = x[:, 1:, :]
        if return_patch_tokens:
            return self.fc_norm(t_tokens)
        return self.fc_norm(t_tokens.mean(1))

    def forward(
        self,
        x: torch.Tensor,
        input_chans: Optional[List[int]] = None,
        return_patch_tokens: bool = False,
        return_all_tokens: bool = False,
    ) -> torch.Tensor:
        x = self.forward_features(
            x, input_chans=input_chans,
            return_patch_tokens=return_patch_tokens,
            return_all_tokens=return_all_tokens,
        )
        return self.head(x)


def labram_base_patch200_200(**kwargs) -> NeuralTransformer:
    """LaBraM-Base. Matches the official ``labram_base_patch200_200`` recipe."""
    model = NeuralTransformer(
        patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4,
        qk_norm=partial(nn.LayerNorm, eps=1e-6),
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# State-dict adapter — braindecode/labram-pretrained → original LaBraM keys
# ─────────────────────────────────────────────────────────────────────────────

def remap_braindecode_state_dict(
    sd: dict, target_time_patches: Optional[int] = None
) -> dict:
    """
    Convert a state_dict from ``braindecode/labram-pretrained`` (HF) to the
    naming scheme used by the upstream LaBraM ``NeuralTransformer``.

    Mapping:
        position_embedding                          → pos_embed
        temporal_embedding                          → time_embed
        patch_embed.temporal_conv.<X>               → patch_embed.<X>
        blocks.<i>.mlp.0.weight/bias                → blocks.<i>.mlp.fc1.weight/bias
        blocks.<i>.mlp.2.weight/bias                → blocks.<i>.mlp.fc2.weight/bias
        norm.weight/bias                            → fc_norm.weight/bias

    If ``target_time_patches`` is given and differs from the source size,
    the time_embed is linearly interpolated to the target length so the
    pretrained 16-position embedding can be reused at e.g. 60-second windows.
    """
    out: dict = {}
    for k, v in sd.items():
        nk = k
        if k == "position_embedding":
            nk = "pos_embed"
        elif k == "temporal_embedding":
            nk = "time_embed"
        elif k.startswith("patch_embed.temporal_conv."):
            nk = "patch_embed." + k[len("patch_embed.temporal_conv."):]
        elif ".mlp.0.weight" in k:
            nk = k.replace(".mlp.0.weight", ".mlp.fc1.weight")
        elif ".mlp.0.bias" in k:
            nk = k.replace(".mlp.0.bias", ".mlp.fc1.bias")
        elif ".mlp.2.weight" in k:
            nk = k.replace(".mlp.2.weight", ".mlp.fc2.weight")
        elif ".mlp.2.bias" in k:
            nk = k.replace(".mlp.2.bias", ".mlp.fc2.bias")
        elif k == "norm.weight":
            nk = "fc_norm.weight"
        elif k == "norm.bias":
            nk = "fc_norm.bias"
        out[nk] = v

    if target_time_patches is not None and "time_embed" in out:
        src = out["time_embed"]                  # (1, S, E)
        if src.shape[1] != target_time_patches:
            # interpolate along time axis
            S, E = src.shape[1], src.shape[2]
            x = src.permute(0, 2, 1)              # (1, E, S)
            x = F.interpolate(x, size=target_time_patches, mode="linear", align_corners=False)
            out["time_embed"] = x.permute(0, 2, 1).contiguous()
            print(f"  [labram] interpolated time_embed: {S} → {target_time_patches}")

    return out
