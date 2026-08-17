"""Core LightningDiT layers, adapted from LightningDiT, DiT, and SiT."""
import math
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from timm.models.vision_transformer import Mlp
from .rms_norm import RMSNorm
from .swiglu import SwiGLUFFN

class MultiHeadCrossAttention(nn.Module):

    def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0, qk_norm=False, fused_attn: bool=True, **block_kwargs):
        super().__init__()
        assert d_model % num_heads == 0, 'd_model must be divisible by num_heads'
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.fused_attn = fused_attn
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x, cond, mask=None):
        (B, N, C) = x.shape
        (B_cond, N_cond, _) = cond.shape
        q = self.q_linear(x)
        k = self.k_linear(cond)
        v = self.v_linear(cond)
        q = q.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B_cond, N_cond, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B_cond, N_cond, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

@torch.compile
def modulate_adasin(x, shift, scale):
    if shift is None:
        return x * (1 + scale.unsqueeze(1))
    return x * (1 + scale) + shift

@torch.compile
def modulate(x, shift, scale):
    if shift is None:
        return x * (1 + scale.unsqueeze(1))
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class Attention(nn.Module):
    """
    Attention module of LightningDiT.
    """

    def __init__(self, dim: int, num_heads: int=8, qkv_bias: bool=False, qk_norm: bool=False, attn_drop: float=0.0, proj_drop: float=0.0, norm_layer: nn.Module=nn.LayerNorm, fused_attn: bool=True, use_rmsnorm: bool=False) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.fused_attn = fused_attn
        if use_rmsnorm:
            norm_layer = RMSNorm
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, rope=None, rope_prefix_len=None) -> torch.Tensor:
        (B, N, C) = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        (q, k, v) = qkv.unbind(0)
        (q, k) = (self.q_norm(q), self.k_norm(k))
        if rope is not None:
            if rope_prefix_len is None or rope_prefix_len == N:
                q = rope(q)
                k = rope(k)
            else:
                prefix = int(rope_prefix_len)
                (q_img, q_extra) = (q[:, :, :prefix], q[:, :, prefix:])
                (k_img, k_extra) = (k[:, :, :prefix], k[:, :, prefix:])
                q = torch.cat([rope(q_img), q_extra], dim=2)
                k = torch.cat([rope(k_img), k_extra], dim=2)
        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    Same as DiT.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int=256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size, bias=True), nn.SiLU(), nn.Linear(hidden_size, hidden_size, bias=True))

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int=10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        Args:
            t: A 1-D Tensor of N indices, one per batch element. These may be fractional.
            dim: The dimension of the output.
            max_period: Controls the minimum frequency of the embeddings.
        Returns:
            An (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class LightningDiTBlock(nn.Module):
    """
    Lightning DiT Block. We add features including: 
    - ROPE
    - QKNorm 
    - RMSNorm
    - SwiGLU
    - No shift AdaLN.
    Not all of them are used in the final model, please refer to the paper for more details.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_qknorm=False, use_swiglu=False, use_rmsnorm=False, wo_shift=False, z_dims=None, dino_concat_self_attention=False, adain_single=True, num_fused_layers=1, encdim_ratio=2, **block_kwargs):
        super().__init__()
        self.adain_single = adain_single
        self.dino_concat_self_attention = bool(dino_concat_self_attention)
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=use_qknorm, use_rmsnorm=use_rmsnorm, **block_kwargs)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = partial(nn.GELU, approximate='tanh')
        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        if self.adain_single:
            self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)
        else:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.z_dims = z_dims
        if self.z_dims is not None and (not self.dino_concat_self_attention):
            self.cross_attn = MultiHeadCrossAttention(d_model=hidden_size, num_heads=num_heads, qk_norm=use_qknorm)

    @torch.compile
    def forward(self, x, c, z=None, feat_rope=None):
        (B, N, C) = x.shape
        if self.adain_single:
            (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = (self.scale_shift_table[None] + c.reshape(B, 6, -1)).chunk(6, dim=1)
        else:
            (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c).chunk(6, dim=1)
        if self.adain_single:
            x_norm = modulate_adasin(self.norm1(x), shift_msa, scale_msa)
            if self.z_dims is not None and self.dino_concat_self_attention and (z is not None):
                xz = torch.cat([x_norm, z], dim=1)
                x = x + gate_msa * self.attn(xz, rope=feat_rope, rope_prefix_len=N)[:, :N]
            else:
                x = x + gate_msa * self.attn(x_norm, rope=feat_rope)
            if self.z_dims is not None and (not self.dino_concat_self_attention):
                x = x + self.cross_attn(x, z)
            x = x + gate_mlp * self.mlp(modulate_adasin(self.norm2(x), shift_mlp, scale_mlp))
            return x
        else:
            x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
            if self.z_dims is not None and self.dino_concat_self_attention and (z is not None):
                xz = torch.cat([x_norm, z], dim=1)
                attn_out = self.attn(xz, rope=feat_rope, rope_prefix_len=N)[:, :N]
            else:
                attn_out = self.attn(x_norm, rope=feat_rope).reshape(B, N, C)
            x = x + gate_msa.unsqueeze(1) * attn_out
            if self.z_dims is not None and (not self.dino_concat_self_attention):
                x = x + self.cross_attn(x, z)
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
            return x

class FinalLayer(nn.Module):
    """
    The final layer of LightningDiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm=False):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        else:
            self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    @torch.compile
    def forward(self, x, c):
        (shift, scale) = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
