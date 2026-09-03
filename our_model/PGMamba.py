"""PG-Mamba model for ultra-widefield OCTA retinal vessel segmentation."""

import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    print("[Warning] mamba_ssm not installed. The model will not run on GPU efficiently.")
    selective_scan_fn = None


class PatchEmbed2D(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape
        x0 = x[:, 0::2, 0::2, :]; x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]; x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, H // 2, W // 2, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim * 2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c',
                      p1=self.dim_scale, p2=self.dim_scale, c=C // self.dim_scale)
        x = self.norm(x)
        return x


class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c',
                      p1=self.dim_scale, p2=self.dim_scale, c=C // self.dim_scale)
        x = self.norm(x)
        return x


class PolarGuidedFusion(nn.Module):
    def __init__(self, num_directions=4, hidden_dim=32):
        super().__init__()
        self.num_directions = num_directions
        self.weight_net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, num_directions),
        )
        nn.init.zeros_(self.weight_net[-1].weight)
        nn.init.zeros_(self.weight_net[-1].bias)
        self._coord_cache = {}

    def get_polar_coords(self, H, W, device):
        cache_key = (H, W, str(device))
        if cache_key in self._coord_cache:
            return self._coord_cache[cache_key]
        cy, cx = H / 2, W / 2
        y = torch.arange(H, device=device, dtype=torch.float32)
        x = torch.arange(W, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        dy = (yy - cy) / (H / 2); dx = (xx - cx) / (W / 2)
        r = torch.sqrt(dy ** 2 + dx ** 2).clamp(max=2.0)
        theta = torch.atan2(dy, dx) / math.pi
        coords = torch.stack([r, theta], dim=-1)
        self._coord_cache[cache_key] = coords
        return coords

    def forward(self, y_directions, H, W):
        B, K, C, L = y_directions.shape
        polar_coords = self.get_polar_coords(H, W, y_directions.device)
        polar_flat = polar_coords.view(L, 2)
        weights = self.weight_net(polar_flat)
        weights = F.softmax(weights, dim=-1)
        weights = weights.T.unsqueeze(0).unsqueeze(2)
        return (y_directions * weights).sum(dim=1)


class SASS_SS2D(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, dt_rank="auto",
                 dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0,
                 dt_init_floor=1e-4, dropout=0., conv_bias=True, bias=False,
                 device=None, dtype=None,
                 use_polar_fusion=0,
                 n_directions=6,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.K = int(n_directions)
        assert self.K in (4, 6), f"n_directions 只支持 4 或 6, got {self.K}"
        self.use_polar_fusion = int(use_polar_fusion)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(self.d_inner, self.d_inner, groups=self.d_inner,
                                bias=conv_bias, kernel_size=d_conv,
                                padding=(d_conv - 1) // 2, **factory_kwargs)
        self.act = nn.SiLU()

        self.x_proj = tuple([
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)])
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = tuple([
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)])
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=self.K, merge=True)
        self.direction_Bs = nn.Parameter(torch.zeros(self.K, self.d_state))
        trunc_normal_(self.direction_Bs, std=0.02)


        if self.use_polar_fusion:
            self.polar_fusion = PolarGuidedFusion(num_directions=self.K, hidden_dim=32)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self._index_cache = {}

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log); A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D); D._no_weight_decay = True
        return D

    def _precompute_diagonal(self, H, W, device):
        indices = []
        for diag_sum in range(H + W - 1):
            coords = []
            for i in range(max(0, diag_sum - W + 1), min(H, diag_sum + 1)):
                coords.append(i * W + (diag_sum - i))
            if diag_sum % 2 == 1:
                coords = coords[::-1]
            indices.extend(coords)
        return torch.tensor(indices, dtype=torch.long, device=device)

    def _precompute_anti_diagonal(self, H, W, device):
        indices = []
        for diag_sum in range(H + W - 1):
            coords = []
            for i in range(max(0, diag_sum - W + 1), min(H, diag_sum + 1)):
                j = W - 1 - (diag_sum - i)
                if 0 <= j < W:
                    coords.append(i * W + j)
            if diag_sum % 2 == 1:
                coords = coords[::-1]
            indices.extend(coords)
        return torch.tensor(indices, dtype=torch.long, device=device)

    def _get_hv_scan_indices(self, H, W, device):

        idx = torch.arange(H * W, device=device).view(H, W)
        o_h = idx.reshape(-1).long()
        o_v = idx.t().reshape(-1).long()
        return o_h, o_v

    def _get_polar_scan_indices(self, H, W, device):

        L = H * W
        cy, cx = H // 2, W // 2
        y = np.arange(H); x = np.arange(W)
        yy, xx = np.meshgrid(y, x, indexing='ij')
        dy = yy - cy; dx = xx - cx
        r = np.sqrt(dy ** 2 + dx ** 2); theta = np.arctan2(dy, dx)
        score_out = r * (L + 1) + (theta + np.pi) * 10
        o_out = np.argsort(score_out.flatten())
        score_in = -r * (L + 1) + (theta + np.pi) * 10
        o_in = np.argsort(score_in.flatten())
        return torch.from_numpy(o_out.copy()).long().to(device), torch.from_numpy(o_in.copy()).long().to(device)

    def get_scan_indices(self, H, W, device):
        cache_key = (H, W, str(device))
        if cache_key in self._index_cache:
            return self._index_cache[cache_key]
        o_h, o_v = self._get_hv_scan_indices(H, W, device)
        o_md = self._precompute_diagonal(H, W, device)
        o_ad = self._precompute_anti_diagonal(H, W, device)

        if self.K == 6:
            o_ro, o_ri = self._get_polar_scan_indices(H, W, device)
            indices_list = [o_h, o_v, o_md, o_ad, o_ro, o_ri]
        elif self.K == 4:
            indices_list = [o_h, o_v, o_md, o_ad]
        else:
            raise ValueError(f"unsupported K={self.K}")
        indices = torch.stack(indices_list, dim=0)
        inverse_indices = torch.stack([torch.argsort(idx) for idx in indices_list], dim=0)
        self._index_cache[cache_key] = (indices, inverse_indices)
        return indices, inverse_indices

    def forward_core(self, x):
        B, C, H, W = x.shape
        L = H * W
        indices, inverse_indices = self.get_scan_indices(H, W, x.device)
        x_flat = x.view(B, C, L)
        xs = torch.stack([x_flat[:, :, indices[k]] for k in range(self.K)], dim=1)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, self.K, -1, L)
        Cs = Cs.float().view(B, self.K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        Bs = Bs + self.direction_Bs.unsqueeze(0).unsqueeze(-1)
        out_y = selective_scan_fn(xs, dts, As, Bs, Cs, Ds, z=None,
                                  delta_bias=dt_projs_bias, delta_softplus=True,
                                  return_last_state=False).view(B, self.K, C, L)
        y_restored = torch.stack([out_y[:, k, :, inverse_indices[k]] for k in range(self.K)], dim=1)

        if self.use_polar_fusion:
            y_fused = self.polar_fusion(y_restored, H, W)
        else:
            y_fused = y_restored.sum(dim=1)
        return y_fused

    def forward(self, x, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y = self.forward_core(x)
        y = y.transpose(1, 2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(self, hidden_dim=0, drop_path=0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 attn_drop_rate=0, d_state=16,
                 use_polar_fusion=0,
                 n_directions=6,
                 **kwargs):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SASS_SS2D(d_model=hidden_dim, dropout=attn_drop_rate,
                                        d_state=d_state, use_polar_fusion=use_polar_fusion,
                                        n_directions=n_directions, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input):
        return input + self.drop_path(self.self_attention(self.ln_1(input)))


class VSSLayer(nn.Module):
    def __init__(self, dim, depth, attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint=False, d_state=16,
                 use_fov_encoder=0, use_polar_fusion=0,
                 n_directions=6,
                 **kwargs):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        self.use_fov_encoder = int(use_fov_encoder)
        if self.use_fov_encoder:
            self.fov_encoder = nn.Sequential(
                nn.Linear(2, dim // 4), nn.GELU(), nn.Linear(dim // 4, dim))
        self._fov_cache = {}
        self.blocks = nn.ModuleList([
            VSSBlock(hidden_dim=dim,
                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                     norm_layer=norm_layer, attn_drop_rate=attn_drop, d_state=d_state,
                     use_polar_fusion=use_polar_fusion,
                     n_directions=n_directions)
            for i in range(depth)])
        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample is not None else None

    def get_fov_coords(self, H, W, device):
        cache_key = (H, W, str(device))
        if cache_key in self._fov_cache:
            return self._fov_cache[cache_key]
        cy, cx = H / 2, W / 2
        y = torch.arange(H, device=device, dtype=torch.float32)
        x = torch.arange(W, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        dy = (yy - cy) / (H / 2); dx = (xx - cx) / (W / 2)
        r = torch.sqrt(dy ** 2 + dx ** 2).clamp(max=2.0)
        theta = torch.atan2(dy, dx) / math.pi
        coords = torch.stack([r, theta], dim=-1)
        self._fov_cache[cache_key] = coords
        return coords

    def forward(self, x):
        B, H, W, C = x.shape
        if self.use_fov_encoder:
            fov_coords = self.get_fov_coords(H, W, x.device).unsqueeze(0)
            x = x + self.fov_encoder(fov_coords)
        for blk in self.blocks:
            x = torch.utils.checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        x_skip = x
        if self.downsample is not None:
            x = self.downsample(x)
        return x, x_skip


class VSSLayer_up(nn.Module):
    def __init__(self, dim, depth, attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 upsample=None, use_checkpoint=False, d_state=16,
                 use_polar_fusion=0,
                 n_directions=6,
                 **kwargs):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        self.upsample = upsample(dim=dim, norm_layer=norm_layer) if upsample is not None else None
        self.blocks = nn.ModuleList([
            VSSBlock(hidden_dim=dim,
                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                     norm_layer=norm_layer, attn_drop_rate=attn_drop, d_state=d_state,
                     use_polar_fusion=use_polar_fusion,
                     n_directions=n_directions)
            for i in range(depth)])

    def forward(self, x, skip=None):
        if self.upsample is not None:
            x = self.upsample(x)
        if skip is not None:
            x = x + skip
        for blk in self.blocks:
            x = torch.utils.checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        return x


class DynamicFOVGating(nn.Module):
    def __init__(self, in_ch, tau=1.15):
        super().__init__()
        self.tau = float(tau)
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_ch + 2, in_ch // 4, 3, 1, 1),
            nn.GroupNorm(8, in_ch // 4), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 4, 1, 3, 1, 1), nn.Sigmoid())

    def get_polar_grid(self, H, W, device):
        y_coord, x_coord = torch.meshgrid(torch.arange(H, device=device),
                                          torch.arange(W, device=device), indexing='ij')
        y_coord = y_coord.float(); x_coord = x_coord.float()
        x_norm = 2 * (x_coord / (W - 1)) - 1
        y_norm = 2 * (y_coord / (H - 1)) - 1
        r = torch.sqrt(x_norm ** 2 + y_norm ** 2)
        theta = torch.atan2(y_norm, x_norm) / math.pi
        return torch.stack([r, theta], dim=0).unsqueeze(0)

    def forward(self, x):
        B, C, H, W = x.shape
        polar_grid = self.get_polar_grid(H, W, x.device).repeat(B, 1, 1, 1)
        r_map = polar_grid[:, 0:1, :, :]
        x_cat = torch.cat([x, polar_grid], dim=1)
        soft_mask = self.gate_conv(x_cat)
        hard_mask = (r_map <= self.tau).float()
        return x * (soft_mask * hard_mask)


class VSSM_Polar(nn.Module):
    def __init__(self, patch_size=4, in_chans=1, num_classes=1,
                 depths=[2, 2, 9, 2], depths_decoder=[2, 9, 2, 2],
                 dims=[96, 192, 384, 768], dims_decoder=[768, 384, 192, 96],
                 d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True, use_checkpoint=False,
                 use_polar_fusion=0, use_fov_gate=1, use_fov_encoder=0,
                 fov_tau=1.15,
                 n_directions=6,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.use_fov_gate = int(use_fov_gate)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i) for i in range(self.num_layers)]
        self.embed_dim = dims[0]; self.num_features = dims[-1]; self.dims = dims

        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans,
                                        embed_dim=self.embed_dim,
                                        norm_layer=norm_layer if patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(VSSLayer(
                dim=dims[i], depth=depths[i],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                use_fov_encoder=use_fov_encoder, use_polar_fusion=use_polar_fusion,
                n_directions=n_directions))

        if self.use_fov_gate:
            self.fov_gate = DynamicFOVGating(dims[-1], tau=fov_tau)

        self.layers_up = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers_up.append(VSSLayer_up(
                dim=dims_decoder[i], depth=depths_decoder[i],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[sum(depths_decoder[:i]):sum(depths_decoder[:i + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if (i != 0) else None,
                use_checkpoint=use_checkpoint, use_polar_fusion=use_polar_fusion,
                n_directions=n_directions))

        self.final_up = Final_PatchExpand2D(dim=dims_decoder[-1], dim_scale=4, norm_layer=norm_layer)
        self.head_seg = nn.Conv2d(dims_decoder[-1] // 4, num_classes, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        skip_list = []
        x = self.patch_embed(x); x = self.pos_drop(x)
        for layer in self.layers:
            x, x_skip = layer(x)
            skip_list.append(x_skip)
        return x, skip_list

    def forward_features_up(self, x, skip_list):
        for inx, layer_up in enumerate(self.layers_up):
            x = layer_up(x, skip=None) if inx == 0 else layer_up(x, skip=skip_list[-(inx + 1)])
        return x

    def forward(self, x):
        x, skip_list = self.forward_features(x)
        if self.use_fov_gate:
            x = x.permute(0, 3, 1, 2)
            x = self.fov_gate(x)
            x = x.permute(0, 2, 3, 1)
        x = self.forward_features_up(x, skip_list)
        x = self.final_up(x)
        x = x.permute(0, 3, 1, 2)
        return torch.sigmoid(self.head_seg(x))


class PGMamba(nn.Module):
    def __init__(self, input_channels=1, num_classes=1,
                 depths=[2, 2, 9, 2], depths_decoder=[2, 9, 2, 2],
                 drop_path_rate=0.2, load_ckpt_path=None,
                 use_polar_fusion=0, use_fov_gate=1, use_fov_encoder=0,
                 use_checkpoint=False, fov_tau=1.15,
                 n_directions=6):
        super().__init__()
        self.num_classes = num_classes
        self.vmunet = VSSM_Polar(
            in_chans=input_channels, num_classes=num_classes,
            depths=depths, depths_decoder=depths_decoder, drop_path_rate=drop_path_rate,
            use_polar_fusion=use_polar_fusion, use_fov_gate=use_fov_gate,
            use_fov_encoder=use_fov_encoder,
            use_checkpoint=use_checkpoint, fov_tau=fov_tau,
            n_directions=n_directions)
        if load_ckpt_path is not None:
            self.load_pretrained(load_ckpt_path)

    def load_pretrained(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        sd = ckpt.get('model', ckpt.get('state_dict', ckpt))
        md = self.state_dict()
        pd = {k: v for k, v in sd.items() if k in md and v.shape == md[k].shape}
        md.update(pd); self.load_state_dict(md)
        print(f"Loaded {len(pd)}/{len(md)} params from {ckpt_path}")

    def forward(self, x):
        return self.vmunet(x)

if __name__ == "__main__":


    if selective_scan_fn is None:
        raise ImportError(
            "mamba_ssm is required to run PG-Mamba. "
            "Please follow docs/ENVIRONMENT.md."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PGMamba(
        input_channels=1, num_classes=1, use_polar_fusion=0,
        use_fov_gate=1, use_fov_encoder=0, fov_tau=1.15, n_directions=6
    ).to(device)

    x = torch.randn(1, 1, 256, 256, device=device)

    model.eval()
    with torch.no_grad():
        y = model(x)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    print("PG-Mamba canonical smoke test")
    print("input :", tuple(x.shape))
    print("output:", tuple(y.shape))
    print("range :", f"[{float(y.min()):.6f}, {float(y.max()):.6f}]")
    print("params:", f"{n_params:.2f} M")