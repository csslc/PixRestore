"""Pixel-space LightningDiT with multi-layer DINO feature guidance."""
from functools import partial

import torch
from einops import repeat
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from timm.models.vision_transformer import Mlp, PatchEmbed
from .layers import FinalLayer, LightningDiTBlock, TimestepEmbedder
from .patch_embed import BottleneckPatchEmbed
from .rope import VisionRotaryEmbeddingFast, broadcat, rotate_half


class LQDegradationEncoder(nn.Module):
    """Tiny LQ-only encoder that gives the DINO layer gate an explicit degradation cue."""

    def __init__(self, hidden_size):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(
            nn.Linear(64, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, lq):
        return self.proj(self.stem(lq).flatten(1))


class LightningDiT_PixelDiffusion(nn.Module):
    """Pixel-space LightningDiT with optional bottleneck patch embedding."""

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=32,
        out_channels=32,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=False,
        use_qknorm=False,
        use_swiglu=False,
        use_rope=False,
        use_rmsnorm=False,
        wo_shift=False,
        use_checkpoint=False,
        z_dims=None,
        encdim_ratio=2,
        adain_single=True,
        num_fused_layers=1,
        pca_dim=None,
        use_dino_layer_router=False,
        dino_layer_router_mode='content',
        same_condition_layer_weights=False,
        gate_temperature=1.0,
        feature_norm='channel_rmsnorm',
        use_bottleneck_patch_embed=True,
        use_lq_degradation_token=False,
        register_local_gate_logit_scale=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.use_rmsnorm = use_rmsnorm
        self.depth = depth
        self.hidden_size = hidden_size
        self.use_checkpoint = use_checkpoint
        self.same_condition_layer_weights = bool(same_condition_layer_weights)
        self.use_dino_layer_router = bool(use_dino_layer_router) and (not self.same_condition_layer_weights)
        if self.use_dino_layer_router:
            mode = str(dino_layer_router_mode or 'content').lower()
            if mode != 'content':
                raise ValueError(f'dino_layer_router_mode only supports content; got {dino_layer_router_mode}')
            self.dino_layer_router_mode = 'content'
        else:
            self.dino_layer_router_mode = 'none'
        self.gate_temperature = max(float(gate_temperature), 1e-06)
        self.feature_norm = str(feature_norm or 'channel_rmsnorm').lower()
        if self.feature_norm not in ('channel_layernorm', 'channel_rmsnorm', 'channel_l2'):
            raise ValueError(
                f'feature_norm must be one of: channel_layernorm, channel_rmsnorm, channel_l2; got {feature_norm}'
            )
        self.use_bottleneck_patch_embed = bool(use_bottleneck_patch_embed)
        self.use_lq_degradation_token = bool(use_lq_degradation_token)
        self.last_dino_teacher_similarity = None
        self.last_dino_layer_router_loss = None
        self.lq_degradation_encoder = (
            LQDegradationEncoder(hidden_size) if self.use_lq_degradation_token else None
        )
        # Present in some released checkpoints; kept for strict load compatibility.
        if bool(register_local_gate_logit_scale):
            self.local_gate_logit_scale = nn.Parameter(torch.zeros(()))

        if self.use_bottleneck_patch_embed:
            self.x_embedder = BottleneckPatchEmbed(
                img_size=input_size,
                patch_size=patch_size,
                in_chans=in_channels,
                pca_dim=pca_dim,
                embed_dim=hidden_size,
                bias=True,
            )
        else:
            try:
                self.x_embedder = PatchEmbed(
                    img_size=input_size,
                    patch_size=patch_size,
                    in_chans=in_channels,
                    embed_dim=hidden_size,
                    bias=True,
                    strict_img_size=False,
                )
            except TypeError:
                self.x_embedder = PatchEmbed(
                    img_size=input_size,
                    patch_size=patch_size,
                    in_chans=in_channels,
                    embed_dim=hidden_size,
                    bias=True,
                )
        self.t_embedder = TimestepEmbedder(hidden_size)
        if self.use_rope:
            half_head_dim = hidden_size // num_heads // 2
            hw_seq_len = input_size // patch_size
            self.feat_rope = VisionRotaryEmbeddingFast(dim=half_head_dim, pt_seq_len=hw_seq_len)
        else:
            self.feat_rope = None
        self.adain_single = adain_single
        if self.adain_single:
            self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.blocks = nn.ModuleList(
            [
                LightningDiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    use_qknorm=use_qknorm,
                    use_swiglu=use_swiglu,
                    use_rmsnorm=use_rmsnorm,
                    wo_shift=wo_shift,
                    z_dims=z_dims,
                    dino_concat_self_attention=False,
                    adain_single=adain_single,
                    num_fused_layers=num_fused_layers,
                    encdim_ratio=encdim_ratio,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, use_rmsnorm=use_rmsnorm)
        self.z_dims = z_dims
        if self.z_dims is not None:
            self.num_fused_layers = num_fused_layers
            approx_gelu = partial(nn.GELU, approximate='tanh')
            self.layer_norm = nn.LayerNorm(self.z_dims)
            self.mlp_ca = Mlp(
                in_features=self.z_dims,
                hidden_features=hidden_size * encdim_ratio,
                out_features=hidden_size,
                act_layer=approx_gelu,
                drop=0,
            )
            if num_fused_layers > 1 and (not self.same_condition_layer_weights):
                self.gate_x_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
                self.gate_c_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
                self.gate_z_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
                if self.use_lq_degradation_token:
                    self.gate_deg_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
                self.gate_block_embed = nn.Embedding(depth, hidden_size)
                gate_extra_tokens = 4 if self.use_lq_degradation_token else 3
                gate_input_dim = hidden_size * (num_fused_layers + gate_extra_tokens)
                self.layer_gates = nn.Linear(gate_input_dim, num_fused_layers)
                if self.use_dino_layer_router:
                    self.dino_prior_content_head = nn.Sequential(nn.SiLU(), nn.Linear(gate_input_dim, num_fused_layers))
        self.initialize_weights()
        if self.use_dino_layer_router:
            self._freeze_exclusive_router_gate_params() # 不再用自适应的 layer_gates，层融合权重改由 content router 头

    def _freeze_exclusive_router_gate_params(self):
        module = getattr(self, 'layer_gates', None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad_(False)

    def initialize_weights(self):

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        if self.use_bottleneck_patch_embed:
            if hasattr(self.x_embedder.proj1, 'weight'):
                w = self.x_embedder.proj1.weight.data
                nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            if hasattr(self.x_embedder.proj2, 'weight'):
                w = self.x_embedder.proj2.weight.data
                nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            if hasattr(self.x_embedder.proj2, 'bias') and self.x_embedder.proj2.bias is not None:
                nn.init.constant_(self.x_embedder.proj2.bias, 0)
        else:
            if hasattr(self.x_embedder.proj, 'weight'):
                w = self.x_embedder.proj.weight.data
                nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            if hasattr(self.x_embedder.proj, 'bias') and self.x_embedder.proj.bias is not None:
                nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        if self.adain_single:
            nn.init.normal_(self.t_block[1].weight, std=0.02)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        if hasattr(self, 'layer_gates'):
            nn.init.constant_(self.layer_gates.weight, 0)
            nn.init.constant_(self.layer_gates.bias, 0)
        if hasattr(self, 'gate_block_embed'):
            nn.init.normal_(self.gate_block_embed.weight, std=0.02)
        if hasattr(self, 'dino_prior_content_head'):
            nn.init.constant_(self.dino_prior_content_head[-1].weight, 0)
            nn.init.constant_(self.dino_prior_content_head[-1].bias, 0)
        if self.lq_degradation_encoder is not None:
            nn.init.normal_(self.lq_degradation_encoder.proj[-1].weight, std=0.02)
            nn.init.constant_(self.lq_degradation_encoder.proj[-1].bias, 0)

    def _train_patch_hw(self):
        (ih, iw) = (int(self.x_embedder.img_size[0]), int(self.x_embedder.img_size[1]))
        (ph, pw) = (int(self.x_embedder.patch_size[0]), int(self.x_embedder.patch_size[1]))
        return (ih // ph, iw // pw)

    def _get_dynamic_rope_hw(self, h_p, w_p, device, dtype):
        if not self.use_rope:
            return None
        (train_h, train_w) = self._train_patch_hw()
        half_head_dim = self.hidden_size // self.num_heads // 2
        theta = 10000
        freqs = 1.0 / theta ** (torch.arange(0, half_head_dim, 2, device=device)[:half_head_dim // 2].float() / half_head_dim)
        t_h = torch.arange(h_p, device=device).float() / max(h_p, 1) * float(train_h)
        t_w = torch.arange(w_p, device=device).float() / max(w_p, 1) * float(train_w)
        freqs_h = torch.einsum('h, f -> h f', t_h, freqs)
        freqs_h = repeat(freqs_h, 'h n -> h (n r)', r=2)
        freqs_w = torch.einsum('w, f -> w f', t_w, freqs)
        freqs_w = repeat(freqs_w, 'w n -> w (n r)', r=2)
        freqs_2d = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim=-1)
        freqs_cos = freqs_2d.cos().reshape(-1, freqs_2d.shape[-1]).to(dtype)
        freqs_sin = freqs_2d.sin().reshape(-1, freqs_2d.shape[-1]).to(dtype)

        def dynamic_rope_fn(t_input):
            return t_input * freqs_cos + rotate_half(t_input) * freqs_sin

        return dynamic_rope_fn

    def unpatchify(self, x, h_p=None, w_p=None):
        c = self.out_channels
        ph = int(self.x_embedder.patch_size[0])
        pw = int(self.x_embedder.patch_size[1])
        if h_p is None or w_p is None:
            h = w = int(x.shape[1] ** 0.5)
            if h * w != x.shape[1]:
                raise ValueError(
                    f'unpatchify: token count {x.shape[1]} is not a perfect square; pass explicit h_p, w_p for rectangular layouts.'
                )
        else:
            (h, w) = (int(h_p), int(w_p))
            if h * w != x.shape[1]:
                raise ValueError(f'unpatchify: h_p*w_p={h * w} != num_tokens={x.shape[1]}')
        x = x.reshape(shape=(x.shape[0], h, w, ph, pw, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], c, h * ph, w * pw))

    def _normalize_feature(self, z):
        if z is None or (not torch.is_tensor(z)) or z.dim() not in (3, 4):
            return z
        if self.feature_norm == 'channel_layernorm':
            return F.layer_norm(z, (z.shape[-1],), eps=1e-06)
        if self.feature_norm == 'channel_rmsnorm':
            rms = z.float().square().mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
            return (z.float() / (rms + 1e-06)).to(dtype=z.dtype)
        return F.normalize(z.float(), p=2.0, dim=-1, eps=1e-06).to(dtype=z.dtype)

    def _normalize_venc_features(self, z):
        if isinstance(z, (list, tuple)):
            return [self._normalize_feature(zi) for zi in z]
        return self._normalize_feature(z)

    def _prepare_venc_features(self, z, normalize=True):
        if self.z_dims is None or z is None:
            return None
        if normalize:
            z = self._normalize_venc_features(z)
        if isinstance(z, (list, tuple)):
            return [self.mlp_ca(self.layer_norm(zi)) for zi in z]
        return self.mlp_ca(self.layer_norm(z))

    def _stack_dino_feature_layers(self, features, dtype, device):
        if features is None:
            return None
        if isinstance(features, (list, tuple)):
            if len(features) != self.num_fused_layers:
                return None
            stacked = torch.stack(features, dim=1)
            if device is None:
                return stacked.to(dtype=dtype)
            return stacked.to(dtype=dtype, device=device)
        if torch.is_tensor(features) and features.dim() == 4:
            if device is None:
                return features.to(dtype=dtype)
            return features.to(dtype=dtype, device=device)
        return None

    def _build_content_teacher_priors(self, z_lq, z_hq):
        lq = self._stack_dino_feature_layers(z_lq, torch.float32, None)
        if lq is None:
            return None
        hq = self._stack_dino_feature_layers(z_hq, torch.float32, lq.device)
        if hq is None or hq.shape != lq.shape:
            return None
        lq = lq.detach().float()
        hq = hq.detach().float()
        diff = lq - hq
        diff_power = diff.square().mean(dim=(-1, -2))
        layer_scale = (0.5 * (lq.square().mean(dim=(-1, -2)) + hq.square().mean(dim=(-1, -2)))).clamp_min(1e-08)
        mse_sim = (1.0 / (1.0 + diff_power / layer_scale)).clamp(0.0, 1.0)
        cos_sim = F.cosine_similarity(lq, hq, dim=-1).mean(dim=-1)
        cos_sim_01 = ((cos_sim + 1.0) * 0.5).clamp(0.0, 1.0)
        similarity = 0.5 * (cos_sim_01 + mse_sim)
        content_prior = torch.softmax(similarity / self.gate_temperature, dim=-1)
        return {'content': content_prior.to(dtype=lq.dtype), 'similarity': similarity}

    def _gate_context(self, z_stack, x, c, block_idx, deg_token=None):
        (B, L, _, _) = z_stack.shape
        z_pool = self.gate_z_norm(z_stack.mean(dim=2))
        x_pool = self.gate_x_norm(x.mean(dim=1))
        if c is None:
            c_pool = torch.zeros_like(x_pool)
        else:
            c_pool = self.gate_c_norm(c.to(dtype=x.dtype))
        block_ids = torch.full((B,), int(block_idx), device=x.device, dtype=torch.long)
        block_pool = self.gate_block_embed(block_ids).to(dtype=x.dtype)
        gate_inputs = [x_pool, c_pool, block_pool]
        if self.use_lq_degradation_token:
            if deg_token is None:
                deg_pool = torch.zeros_like(x_pool)
            else:
                deg_pool = self.gate_deg_norm(deg_token.to(dtype=x.dtype))
            gate_inputs.append(deg_pool)
        gate_inputs.append(z_pool.flatten(1))
        return torch.cat(gate_inputs, dim=-1)

    def _predict_shared_dino_layer_router(self, z, x, c, deg_token=None):
        if not self.use_dino_layer_router or not isinstance(z, (list, tuple)) or len(z) <= 1:
            return None
        z_stack = torch.stack(z, dim=1)
        gate_context = self._gate_context(z_stack, x, c, block_idx=0, deg_token=deg_token)
        content_logits = self.dino_prior_content_head(gate_context)
        return torch.softmax(content_logits.float(), dim=-1).to(dtype=content_logits.dtype).clamp_min(1e-08)

    def _gate_venc_features(
        self,
        z,
        x,
        c=None,
        block_idx=0,
        deg_token=None,
        z_teacher=None,
        z_hq_teacher=None,
        shared_prior_probs=None,
        compute_prior_loss=True,
    ):
        if not isinstance(z, (list, tuple)):
            return z
        if len(z) == 1:
            return z[0]
        if len(z) != self.num_fused_layers:
            raise ValueError(f'Expected {self.num_fused_layers} visual feature layers, got {len(z)}.')
        z_stack = torch.stack(z, dim=1)
        (B, L, N_z, _) = z_stack.shape

        if self.same_condition_layer_weights:
            return z_stack.mean(dim=1)

        gate_context = self._gate_context(z_stack, x, c, block_idx, deg_token=deg_token)
        if self.use_dino_layer_router:
            prior_probs = shared_prior_probs
            if prior_probs is None:
                content_logits = self.dino_prior_content_head(gate_context)
                prior_probs = torch.softmax(content_logits.float(), dim=-1).to(dtype=content_logits.dtype).clamp_min(1e-08)
            weights = prior_probs.to(dtype=z_stack.dtype)
        else:
            prior_probs = None
            weights = self.layer_gates(gate_context).softmax(dim=-1).to(dtype=z_stack.dtype)

        if self.use_dino_layer_router and compute_prior_loss and prior_probs is not None:
            teacher_priors = self._build_content_teacher_priors(
                z_teacher if z_teacher is not None else z_stack,
                z_hq_teacher,
            )
            if teacher_priors is not None:
                teacher_weights = teacher_priors['content']
                self.last_dino_teacher_similarity = teacher_priors['similarity'].detach()
                prior_loss = -(teacher_weights.float() * prior_probs.float().clamp_min(1e-08).log()).sum(dim=-1).mean()
                self._dino_layer_router_losses.append(prior_loss)

        token_weights = weights[:, None, :].expand(B, N_z, L)
        return (z_stack * token_weights.permute(0, 2, 1)[:, :, :, None]).sum(dim=1)

    def enable_fused_attn(self):
        for block in self.blocks:
            block.attn.fused_attn = True
            if hasattr(block, 'cross_attn') and block.cross_attn is not None:
                block.cross_attn.fused_attn = True

    def disable_fused_attn(self):
        for block in self.blocks:
            block.attn.fused_attn = False
            if hasattr(block, 'cross_attn') and block.cross_attn is not None:
                block.cross_attn.fused_attn = False

    def forward(self, x, t, r=None, z=None, z_hq=None):
        use_checkpoint = self.use_checkpoint
        self.last_dino_layer_router_loss = None
        self.last_dino_teacher_similarity = None
        self._dino_layer_router_losses = []
        (N, C, H, W) = x.shape
        (ph, pw) = (int(self.x_embedder.patch_size[0]), int(self.x_embedder.patch_size[1]))
        if H % ph != 0 or W % pw != 0:
            raise ValueError(f'Input spatial size ({H}, {W}) must be divisible by patch_size ({ph}, {pw}).')
        (h_p, w_p) = (H // ph, W // pw)
        lq_image = x[:, :3] if self.lq_degradation_encoder is not None else None
        x = self.x_embedder(x)
        (N, T, D) = x.shape
        if T != h_p * w_p:
            raise RuntimeError(f'embed tokens T={T} != h_p*w_p={h_p * w_p}')
        t = self.t_embedder(t)
        c = t
        if self.adain_single:
            c0 = self.t_block(c)
        deg_token = self.lq_degradation_encoder(lq_image) if lq_image is not None else None
        z_teacher = self._normalize_venc_features(z)
        z_hq_teacher = self._normalize_venc_features(z_hq)
        z = self._prepare_venc_features(z_teacher, normalize=False)
        (train_h, train_w) = self._train_patch_hw()
        if self.use_rope and h_p == train_h and (w_p == train_w):
            feat_rope = self.feat_rope
        elif self.use_rope:
            feat_rope = self._get_dynamic_rope_hw(h_p, w_p, x.device, x.dtype)
        else:
            feat_rope = None
        shared_prior_probs = self._predict_shared_dino_layer_router(z, x, c, deg_token=deg_token)
        for (i, block) in enumerate(self.blocks):
            z_block = self._gate_venc_features(
                z,
                x,
                c,
                i,
                deg_token=deg_token,
                z_teacher=z_teacher,
                z_hq_teacher=z_hq_teacher,
                shared_prior_probs=shared_prior_probs,
                compute_prior_loss=i == 0 or shared_prior_probs is None,
            )
            if use_checkpoint:
                x = checkpoint(block, x, c0 if self.adain_single else c, z_block, feat_rope, use_reentrant=True)
            else:
                x = block(x, c0 if self.adain_single else c, z_block, feat_rope)
        if self._dino_layer_router_losses:
            self.last_dino_layer_router_loss = torch.stack(self._dino_layer_router_losses).mean()
        x = self.final_layer(x, c)
        x = self.unpatchify(x, h_p, w_p)
        return (x, [])
