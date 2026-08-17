"""Pixel-space MeanFlow training objective and sampler."""
import random
from typing import Optional, Sequence, Union
import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

def stopgrad(x):
    return x.detach().clone()

def _valid_loss_mask(invalid_mask: Optional[torch.Tensor], target: torch.Tensor) -> Optional[torch.Tensor]:
    if invalid_mask is None:
        return None
    mask = invalid_mask.to(device=target.device)
    if mask.dim() == target.dim() - 1:
        mask = mask.unsqueeze(1)
    if mask.dim() != target.dim():
        raise ValueError(f'invalid_mask must have shape (B,H,W), (B,1,H,W), or (B,C,H,W); got {tuple(mask.shape)}')
    if mask.shape[0] != target.shape[0]:
        raise ValueError(f'invalid_mask batch size {mask.shape[0]} != target batch size {target.shape[0]}')
    if mask.shape[-2:] != target.shape[-2:]:
        mask = F.interpolate(mask.float(), size=target.shape[-2:], mode='nearest')
    else:
        mask = mask.float()
    if mask.shape[1] == 1 and target.shape[1] != 1:
        mask = mask.expand(-1, target.shape[1], -1, -1)
    elif mask.shape[1] != target.shape[1]:
        raise ValueError(f'invalid_mask channels {mask.shape[1]} != target channels {target.shape[1]}')
    return (1.0 - mask.clamp(0.0, 1.0)).to(dtype=target.dtype)

def _masked_mean_square(error: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if valid_mask is None:
        return (error ** 2).mean()
    return (error ** 2 * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)

def _as_feature_list(features: Union[torch.Tensor, Sequence[torch.Tensor]]) -> Sequence[torch.Tensor]:
    if isinstance(features, torch.Tensor):
        return [features]
    return list(features)

def _mean_non_batch(x: torch.Tensor) -> torch.Tensor:
    if x.dim() <= 1:
        return x
    return x.mean(dim=tuple(range(1, x.dim())))

def _get_model_attr(model: nn.Module, name: str, default=None):
    value = getattr(model, name, default)
    if value is not default:
        return value
    wrapped = getattr(model, 'module', None)
    if wrapped is not None:
        return getattr(wrapped, name, default)
    return default

def _resolve_hierar_loss_layer_weights(model: nn.Module, args) -> Optional[torch.Tensor]:
    """Layer weights for hierarchical DINO feature loss.

    - ``average_hierar_loss_weights``: equal mean over layers (return None).
    - ``use_dino_layer_router``: reverse-weight by teacher similarity
      ``(1 - sim) / tau`` then softmax (high sim → low weight).
    - otherwise: equal mean (return None).
    """
    if getattr(args, 'average_hierar_loss_weights', False):
        return None
    if not getattr(args, 'use_dino_layer_router', False):
        return None
    dino_similarity = _get_model_attr(model, 'last_dino_teacher_similarity', None)
    if dino_similarity is None:
        return None
    tau = max(float(getattr(args, 'gate_temperature', 1.0) or 1.0), 1e-06)
    sim = dino_similarity.detach().float()
    logits = (1.0 - sim) / tau
    return torch.softmax(logits, dim=-1)

def _multi_layer_feature_loss(pred_features: Union[torch.Tensor, Sequence[torch.Tensor]], target_features: Union[torch.Tensor, Sequence[torch.Tensor]], loss_type: str='cosine', layer_weights: Optional[torch.Tensor]=None) -> torch.Tensor:
    pred_list = _as_feature_list(pred_features)
    target_list = _as_feature_list(target_features)
    if len(pred_list) != len(target_list):
        raise ValueError(f'DINO feature loss expects same layer count, got {len(pred_list)} and {len(target_list)}')
    per_layer_loss_maps = []
    for (pred, target) in zip(pred_list, target_list):
        target = target.detach().to(device=pred.device, dtype=pred.dtype)
        if pred.shape != target.shape:
            raise ValueError(f'DINO feature shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}')
        if loss_type == 'mse':
            loss_map = (pred.float() - target.float()).square().mean(dim=-1)
        else:
            pred_n = F.normalize(pred.float(), dim=-1)
            target_n = F.normalize(target.float(), dim=-1)
            loss_map = 1.0 - (pred_n * target_n).sum(dim=-1)
        per_layer_loss_maps.append(loss_map.reshape(loss_map.shape[0], -1))
    losses = torch.stack(per_layer_loss_maps, dim=-1)
    if layer_weights is None:
        return losses.mean()
    weights = layer_weights.detach().to(device=losses.device, dtype=losses.dtype)
    if weights.dim() == 1:
        weights = weights.unsqueeze(0)
    if weights.shape[-1] != losses.shape[-1]:
        raise ValueError(f'DINO feature loss layer weights expect {losses.shape[-1]} layers, got {weights.shape[-1]}')
    if weights.dim() == 2:
        if weights.shape[0] == 1 and losses.shape[0] != 1:
            weights = weights.expand(losses.shape[0], -1)
        if weights.shape != (losses.shape[0], losses.shape[-1]):
            raise ValueError(f'DINO feature loss global weights shape {tuple(weights.shape)} != {(losses.shape[0], losses.shape[-1])}')
        weights = weights[:, None, :].expand_as(losses)
    elif weights.dim() == 3:
        if weights.shape != losses.shape:
            raise ValueError(f'DINO feature loss spatial weights shape {tuple(weights.shape)} != losses {tuple(losses.shape)}')
    else:
        raise ValueError(f'DINO feature loss weights must be [B,L] or [B,N,L], got {tuple(weights.shape)}')
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-08)
    return (losses * weights).sum(dim=-1).mean()

class PixelDiffusion(nn.Module):
    """
    Loss and Evaluation for Pixel Diffusion - 预测x0后转为v
    """

    def __init__(self, flow_ratio=0.5, time_dist=['uniform', 0, 1.0], z_start='noise', alpha=0.9, delta_t=0.001, cfg_ratio=0.1, cfg_scale=2.0, image_size=32, channels=3, weak_cond_strength=0.3, interp_type='lin', uncond_type='zero', cfg_uncond='u', norm_p: float=1.0, norm_eps: float=0.01, u_weight=1, a=1.0, b=1.0, accelerator=None, train_steps=2, z_dim=768, z_length=256, t_start: float=0.0, t_end: float=1.0, use_cos=False, args=None):
        super().__init__()
        self.delta_t = delta_t
        self.flow_ratio = flow_ratio
        self.time_dist = time_dist
        self.alpha = alpha
        self.z_start = z_start
        self.interp_type = interp_type
        self.uncond_type = uncond_type
        self.cfg_ratio = cfg_ratio
        self.cfg_scale = cfg_scale
        self.cfg_uncond = cfg_uncond
        self.args = args
        self.u_weight = u_weight
        self.norm_p = norm_p
        self.norm_eps = norm_eps
        self.device = accelerator.device if accelerator is not None else torch.device('cuda')
        self.a = a
        self.b = b
        self.train_steps = train_steps
        self.t_start = t_start
        self.t_end = t_end
        self.use_cos = use_cos
        self.last_hierar_loss = None

    def interpolate(self, cond, uncond, alpha, interp_type='linear'):
        """
        插值函数
        """
        if interp_type == 'sph':
            return alpha * cond + (1 - alpha ** 2) ** 0.5 * uncond
        elif interp_type == 'lin':
            return alpha * cond + (1 - alpha) * uncond

    def get_time_shift_factor(self, resolution, base_resolution=1024):
        """
        计算time shift因子 α = √(m/n)
        """
        if not hasattr(self.args, 'use_time_shift') or not self.args.use_time_shift:
            return 1.0
        alpha = np.sqrt(resolution / (base_resolution * base_resolution))
        return alpha

    def apply_time_shift(self, t, alpha):
        """
        应用time shift: t_m = (α * t_n) / (1 + (α - 1) * t_n)
        """
        if alpha == 1.0:
            return t
        return alpha * t / (1 + (alpha - 1) * t)

    def sample_t_r(self, B, device):
        """
        采样时间步t和r
        """
        (dist, *params) = self.time_dist
        if dist == 'uniform':
            samples = torch.rand(B, 2, device=device)
        else:
            (mu, sigma) = params
            normal = torch.randn(B, 2, device=device) * sigma + mu
            samples = 1 / (1 + torch.exp(-normal))
        (t, r) = (samples[:, 0], samples[:, 1])
        (t, r) = (torch.maximum(t, r), torch.minimum(t, r))
        num_eq = int(self.flow_ratio * B)
        idx = torch.randperm(B, device=device)[:num_eq]
        r[idx] = t[idx]
        return (t, r)

    def loss_fm(self, model, lq, hq, z=None, weight_dtype=None, invalid_mask=None, z_hq=None, dino_loss_fn=None):
        """
        Pixel Diffusion loss function: predict x0, and then calculate v for loss function
        
        Args:
            model: input (z, t), output x0 prediction
            lq: (B, C, H, W)
            hq: (B, C, H, W)
            z: condition, optinal
            invalid_mask: optional, 1 denotes the invalid area in depth is masked. Invalid area is not considered in loss function.
        
        Returns:
            loss, loss_backward, None
        """
        self.last_hierar_loss = None
        (B, device) = (hq.size(0), hq.device)
        valid_loss_mask = _valid_loss_mask(invalid_mask, hq)
        fixed_train_t = getattr(self.args, 'fixed_train_t', None)
        if fixed_train_t is not None:
            t = torch.full((B,), float(fixed_train_t), device=device)
        elif self.time_dist[0] == 'uniform':
            t = torch.rand(B, device=device)
        elif self.time_dist[0] == 'lognorm':
            (mu, sigma) = (self.time_dist[-2], self.time_dist[-1])
            rnd_normal = torch.randn(B, device=device)
            t = torch.sigmoid(rnd_normal * sigma + mu)
        t_shifted = t
        (H, W) = hq.shape[-2:]
        C = hq.shape[1]
        base_dimension = getattr(self.args, 'time_shift_base_dimension', 4096)
        alpha = np.sqrt(H * W * C / base_dimension)
        if self.args.use_time_shift:
            t_shifted = self.apply_time_shift(t, alpha)
        t_shifted_ = rearrange(t_shifted, 'b -> b 1 1 1')
        cond_strength_aelq = torch.sigmoid(torch.randn(B, device=device, dtype=hq.dtype) * self.args.cond_strength_aelq_list[1] + self.args.cond_strength_aelq_list[0])
        cond_strength_aelq = rearrange(cond_strength_aelq, 'b -> b 1 1 1')
        if self.args.aug_noise:
            eps = alpha * torch.randn_like(lq)
        else:
            eps = torch.randn_like(lq)
        if self.cfg_ratio > 0:
            cfg_mask = torch.rand(B, device=device) < self.cfg_ratio
            cfg_indices = (cfg_mask > 0).nonzero(as_tuple=True)[0]
            if self.args.use_aelq:
                weak_cond_strength_aelq = random.uniform(self.args.weak_cond_strength_aelq_list[0], self.args.weak_cond_strength_aelq_list[1])
                lq_weak = self.interpolate(lq, torch.zeros_like(lq), weak_cond_strength_aelq, self.interp_type)
                lq_mixed = self.interpolate(lq, torch.randn_like(lq), cond_strength_aelq, interp_type='sph')
                lq_mixed[cfg_indices] = lq_weak[cfg_indices]
            else:
                lq_mixed = torch.zeros_like(lq)
            if self.args.use_venc:
                if isinstance(z, torch.Tensor):
                    z_weak = self.interpolate(z, torch.zeros_like(z), self.args.weak_cond_strength_venc, self.interp_type)
                    z_mixed = self.interpolate(z, torch.randn_like(z), self.args.cond_strength_venc, interp_type='sph')
                    z_mixed[cfg_indices] = z_weak[cfg_indices]
                elif isinstance(z, (list, tuple)):
                    z_weak = [self.interpolate(z_i, torch.zeros_like(z_i), self.args.weak_cond_strength_venc, self.interp_type) for z_i in z]
                    z_mixed = [self.interpolate(z_i, torch.randn_like(z_i), self.args.cond_strength_venc, interp_type='sph') for z_i in z]
                    for i in range(len(z)):
                        z_mixed[i][cfg_indices] = z_weak[i][cfg_indices]
                else:
                    raise TypeError(f'Unsupported type for z: {type(z)}')
            else:
                z_mixed = None
        else:
            if self.args.use_aelq:
                lq_mixed = self.interpolate(lq, torch.randn_like(lq), cond_strength_aelq, interp_type='sph')
            else:
                lq_mixed = torch.randn_like(lq)
            if self.args.use_venc:
                if isinstance(z, torch.Tensor):
                    z_mixed = self.interpolate(z, torch.randn_like(z), self.args.cond_strength_venc, interp_type='sph')
                elif isinstance(z, (list, tuple)):
                    z_mixed = [self.interpolate(z_i, torch.randn_like(z_i), self.args.cond_strength_venc, interp_type='sph') for z_i in z]
                else:
                    raise TypeError(f'Unsupported type for z: {type(z)}')
            else:
                z_mixed = None
        z_t = (1 - t_shifted_) * hq + t_shifted_ * eps
        v = (z_t - hq) / t_shifted_.clamp_min(self.args.t_eps)
        inp = torch.cat([lq_mixed, z_t], dim=1)
        if z_hq is not None:
            (x_pred, x_fea) = model(inp, t, z=z_mixed, z_hq=z_hq)
        else:
            (x_pred, x_fea) = model(inp, t, z=z_mixed)
        v_pred = (z_t - x_pred) / t_shifted_.clamp_min(self.args.t_eps)
        v_error = v_pred - v
        loss = _masked_mean_square(v_error, valid_loss_mask)
        dino_layer_router_loss = getattr(model, 'last_dino_layer_router_loss', None)
        if getattr(self.args, 'use_dino_layer_router', False) and (not getattr(self.args, 'same_condition_layer_weights', False)) and dino_layer_router_loss is not None:
            loss = loss + dino_layer_router_loss * float(getattr(self.args, 'dino_layer_router_loss_weight', 0.01))
        if getattr(self.args, 'use_hierar_loss', False) and dino_loss_fn is not None and (z_hq is not None):
            pred_dino = dino_loss_fn(x_pred.clamp(-1.0, 1.0))
            hierar_layer_weights = _resolve_hierar_loss_layer_weights(model, self.args)
            hierar_loss = _multi_layer_feature_loss(pred_dino, z_hq, loss_type=str(getattr(self.args, 'dino_hierar_loss_type', 'cosine') or 'cosine').lower(), layer_weights=hierar_layer_weights)
            self.last_hierar_loss = hierar_loss
            loss = loss + hierar_loss * float(getattr(self.args, 'dino_hierar_loss_weight', 0.03))
        if self.use_cos:
            cos_loss = 1 - F.cosine_similarity(v_pred, v, dim=1)
            if valid_loss_mask is not None:
                cos_valid_mask = valid_loss_mask[:, :1].squeeze(1)
                cos_loss = (cos_loss * cos_valid_mask).sum() / cos_valid_mask.sum().clamp_min(1.0)
            else:
                cos_loss = cos_loss.mean()
            loss = loss + cos_loss
        adp_wt = (loss + self.norm_eps) ** self.norm_p
        loss_backward = loss / adp_wt.detach()
        return (loss, loss_backward, None)

    @torch.no_grad()
    def sample_multistep_fm(self, model, lq, venc_fea=None, n_steps: int=25, schedule: str='linear'):
        """
        Pixel Diffusion Evaluation Function
        
        Args:
            model: input (z, t), output x0 prediction
            lq: (B, C, H, W)
            venc_fea: optional
            n_steps: inference steps
            schedule
        Returns:
            hq_hat: Restored image (B, C, H, W)
        """
        (B, device) = (lq.size(0), lq.device)
        lq_weak = None
        venc_fea_weak = None
        use_cfg = self.cfg_ratio > 0 and int(n_steps) > 1
        if self.args.aug_noise:
            (H, W) = lq.shape[-2:]
            C = lq.shape[1]
            base_dimension = getattr(self.args, 'time_shift_base_dimension', 4096)
            alpha = np.sqrt(H * W * C / base_dimension)
            eps = alpha * torch.randn_like(lq)
        else:
            eps = torch.randn_like(lq)
        if use_cfg:
            if self.args.use_aelq:
                weak_cond_strength_aelq = (self.args.weak_cond_strength_aelq_list[0] + self.args.weak_cond_strength_aelq_list[1]) / 2.0
                lq_weak = self.interpolate(lq, torch.zeros_like(lq), weak_cond_strength_aelq, self.interp_type)
                lq = self.interpolate(lq, torch.randn_like(lq), self.args.cond_strength_aelq_test, interp_type='sph') if hasattr(self.args, 'cond_strength_aelq_test') else lq
            else:
                lq_weak = torch.zeros_like(lq)
                lq = torch.randn_like(lq)
            if self.args.use_venc:
                if isinstance(venc_fea, (list, tuple)):
                    venc_fea_weak = [self.interpolate(fea, torch.zeros_like(fea), self.args.weak_cond_strength_venc, self.interp_type) for fea in venc_fea]
                    venc_fea = [self.interpolate(fea, torch.randn_like(fea), self.args.cond_strength_venc, interp_type='sph') for fea in venc_fea]
                else:
                    venc_fea_weak = self.interpolate(venc_fea, torch.zeros_like(venc_fea), self.args.weak_cond_strength_venc, self.interp_type)
                    venc_fea = self.interpolate(venc_fea, torch.randn_like(venc_fea), self.args.cond_strength_venc, interp_type='sph')
            else:
                venc_fea_weak = None
        else:
            if self.args.use_aelq:
                lq = self.interpolate(lq, torch.randn_like(lq), self.args.cond_strength_aelq_test, interp_type='sph')
            else:
                lq = torch.randn_like(lq)
            if self.args.use_venc:
                if isinstance(venc_fea, (list, tuple)):
                    venc_fea = [self.interpolate(fea, torch.randn_like(fea), self.args.cond_strength_venc, interp_type='sph') for fea in venc_fea]
                else:
                    venc_fea = self.interpolate(venc_fea, torch.randn_like(venc_fea), self.args.cond_strength_venc, interp_type='sph')
        z = eps
        if schedule == 'linear':
            t_seq = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
        elif schedule == 'cosine':
            s = torch.linspace(0, np.pi / 2, n_steps + 1, device=device)
            t_seq = torch.cos(s)
        else:
            raise ValueError("schedule must be 'linear' or 'cosine'")
        for i in range(n_steps):
            (t_cur, t_next) = (t_seq[i], t_seq[i + 1])
            dt = t_cur - t_next
            model_inp = torch.cat([lq, z], dim=1)
            model_t = t_cur.repeat(B)
            model_venc_fea = venc_fea
            if use_cfg and t_cur <= self.t_end and (t_cur >= self.t_start):
                inp_weak = torch.cat([lq_weak, z], dim=1)
                model_inp = torch.cat([model_inp, inp_weak], dim=0)
                model_t = model_t.repeat(2)
                if venc_fea is not None:
                    if isinstance(venc_fea, (list, tuple)):
                        model_venc_fea = [torch.cat([v, w], dim=0) for (v, w) in zip(venc_fea, venc_fea_weak)]
                    else:
                        model_venc_fea = torch.cat([venc_fea, venc_fea_weak], dim=0)
            (x_pred, _) = model(model_inp, model_t, z=model_venc_fea)
            t_cur_ = rearrange(t_cur, ' -> 1 1 1 1')
            if use_cfg and t_cur <= self.t_end and (t_cur >= self.t_start):
                (x_pred_cond, x_pred_weak) = x_pred.chunk(2)
                v_pred_cond = (z - x_pred_cond) / t_cur_.clamp_min(self.args.t_eps)
                v_pred_weak = (z - x_pred_weak) / t_cur_.clamp_min(self.args.t_eps)
                v_pred = v_pred_weak + self.cfg_scale * (v_pred_cond - v_pred_weak)
            else:
                v_pred = (z - x_pred) / t_cur_.clamp_min(self.args.t_eps)
            z = z - dt * v_pred
        hq_hat = z
        return hq_hat
