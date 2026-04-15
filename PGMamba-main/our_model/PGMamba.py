import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from functools import partial
from typing import Optional, Callable
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# 尝试导入 Mamba 核心算子
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except ImportError:
    print("[Warning] mamba_ssm not installed. The model will not run on GPU efficiently.")
    selective_scan_fn = None

# ==============================================================================
# 1. 基础组件 (Patch Embeddings & Merging)
# ==============================================================================

class PatchEmbed2D(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)  # B C H W -> B H W C
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
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
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


# ==============================================================================
# 2. 极坐标引导融合模块 (PolarGuidedFusion)
# ==============================================================================

class PolarGuidedFusion(nn.Module):
    """
    根据像素的极坐标位置，自适应地为不同扫描方向分配权重
    
    直觉：
    - 靠近中心的像素：径向扫描更重要（血管从中心辐射）
    - FOV 边缘的像素：对角线扫描可能更重要（切向血管）
    """
    def __init__(self, num_directions=4, hidden_dim=32):
        super().__init__()
        self.num_directions = num_directions
        
        # 从极坐标 (r, θ) 生成方向权重
        self.weight_net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_directions),
        )
        
        # 初始化：让初始权重接近均匀分布
        nn.init.zeros_(self.weight_net[-1].weight)
        nn.init.zeros_(self.weight_net[-1].bias)
        
        # 缓存极坐标网格
        self._coord_cache = {}
    
    def get_polar_coords(self, H, W, device):
        """生成归一化的极坐标网格"""
        cache_key = (H, W, str(device))
        if cache_key in self._coord_cache:
            return self._coord_cache[cache_key]
        
        cy, cx = H / 2, W / 2
        y = torch.arange(H, device=device, dtype=torch.float32)
        x = torch.arange(W, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        # 归一化到 [-1, 1]
        dy = (yy - cy) / (H / 2)
        dx = (xx - cx) / (W / 2)
        
        # 极坐标
        r = torch.sqrt(dy**2 + dx**2).clamp(max=2.0)  # 半径 [0, ~1.4]
        theta = torch.atan2(dy, dx) / math.pi          # 角度 [-1, 1]
        
        coords = torch.stack([r, theta], dim=-1)  # (H, W, 2)
        self._coord_cache[cache_key] = coords
        return coords

    def forward(self, y_directions, H, W):
        """
        Args:
            y_directions: (B, K, C, L) - K个方向的输出
            H, W: 空间尺寸
        Returns:
            y_fused: (B, C, L) - 加权融合后的输出
        """
        B, K, C, L = y_directions.shape
        
        # 生成极坐标网格
        polar_coords = self.get_polar_coords(H, W, y_directions.device)  # (H, W, 2)
        polar_flat = polar_coords.view(L, 2)  # (L, 2)
        
        # 生成每个位置的方向权重
        weights = self.weight_net(polar_flat)      # (L, K)
        weights = F.softmax(weights, dim=-1)       # 归一化
        weights = weights.T.unsqueeze(0).unsqueeze(2)  # (1, K, 1, L)
        
        # 加权融合
        y_fused = (y_directions * weights).sum(dim=1)  # (B, C, L)
        
        return y_fused


# ==============================================================================
# 3. SASS_SS2D (4方向: 对角线×2 + 径向×2)
# ==============================================================================

class SASS_SS2D(nn.Module):
    """
    Spatially-Aware State Space 2D
    
    4方向扫描设计:
    - o1: 主对角线蛇形 (捕获 45° 走向血管)
    - o2: 副对角线蛇形 (捕获 135° 走向血管)
    - o3: 径向向外 (从中心向边缘辐射的血管)
    - o4: 径向向内 (从边缘向中心汇聚的血管)
    """
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        
        # 固定 4 方向
        self.K = 4

        # 1. 基础投影层
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        # 2. 状态空间投影（4 个方向）
        self.x_proj = tuple([
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ])
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = tuple([
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ])
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        # 3. 初始化 SSM 参数
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=self.K, merge=True)
        
        # 方向偏置（4 个方向）
        self.direction_Bs = nn.Parameter(torch.zeros(self.K, self.d_state))
        trunc_normal_(self.direction_Bs, std=0.02)

        # 4. 极坐标引导融合
        self.polar_fusion = PolarGuidedFusion(num_directions=self.K, hidden_dim=32)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        
        # 缓存索引
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
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    # =========================================================================
    # 扫描索引生成
    # =========================================================================
    
    def _precompute_diagonal(self, H, W, device):
        """主对角线扫描索引（蛇形）"""
        indices = []
        for diag_sum in range(H + W - 1):
            coords = []
            for i in range(max(0, diag_sum - W + 1), min(H, diag_sum + 1)):
                j = diag_sum - i
                coords.append(i * W + j)
            # 蛇形：奇数对角线反向
            if diag_sum % 2 == 1:
                coords = coords[::-1]
            indices.extend(coords)
        return torch.tensor(indices, dtype=torch.long, device=device)
    
    def _precompute_anti_diagonal(self, H, W, device):
        """副对角线扫描索引（蛇形）"""
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

    def _get_polar_scan_indices(self, H, W, device):
        """生成径向扫描索引（从外到内 & 从内到外）"""
        L = H * W
        cy, cx = H // 2, W // 2
        
        # 生成坐标网格
        y = np.arange(H)
        x = np.arange(W)
        yy, xx = np.meshgrid(y, x, indexing='ij')
        
        # 极坐标
        dy = yy - cy
        dx = xx - cx
        r = np.sqrt(dy**2 + dx**2)
        theta = np.arctan2(dy, dx)
        
        # 径向向外（从中心向边缘）
        score_out = r * (L + 1) + (theta + np.pi) * 10
        o_out = np.argsort(score_out.flatten())
        
        # 径向向内（从边缘向中心）
        score_in = -r * (L + 1) + (theta + np.pi) * 10
        o_in = np.argsort(score_in.flatten())
        
        return (
            torch.from_numpy(o_out.copy()).long().to(device),
            torch.from_numpy(o_in.copy()).long().to(device)
        )

    def get_scan_indices(self, H, W, device):
        """
        生成 4 方向扫描索引:
        - o1: 主对角线蛇形
        - o2: 副对角线蛇形
        - o3: 径向向外
        - o4: 径向向内
        """
        cache_key = (H, W, str(device))
        if cache_key in self._index_cache:
            return self._index_cache[cache_key]
        
        # 对角线方向
        o1 = self._precompute_diagonal(H, W, device)
        o2 = self._precompute_anti_diagonal(H, W, device)
        
        # 径向方向
        o3, o4 = self._get_polar_scan_indices(H, W, device)
        
        indices_list = [o1, o2, o3, o4]
        
        # 计算逆索引
        indices = torch.stack(indices_list, dim=0)
        inverse_indices = torch.stack([torch.argsort(idx) for idx in indices_list], dim=0)
        
        self._index_cache[cache_key] = (indices, inverse_indices)
        return indices, inverse_indices

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W

        # 1. 获取索引
        indices, inverse_indices = self.get_scan_indices(H, W, x.device)
        
        # 2. 扫描：按 4 种路径重排
        x_flat = x.view(B, C, L)
        xs = torch.stack([
            x_flat[:, :, indices[k]] for k in range(self.K)
        ], dim=1)  # (B, K, C, L)
        
        # 3. 投影
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)
        
        # 4. 准备 SSM 参数
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, self.K, -1, L)
        Cs = Cs.float().view(B, self.K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        
        # 加入方向偏置
        Bs = Bs + self.direction_Bs.unsqueeze(0).unsqueeze(-1)
        
        # 5. Selective Scan
        out_y = selective_scan_fn(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, self.K, C, L)

        # 6. 逆扫描还原
        y_restored = torch.stack([
            out_y[:, k, :, inverse_indices[k]] for k in range(self.K)
        ], dim=1)  # (B, K, C, L)
        
        # 7. 极坐标引导融合（替代简单求和）
        y_fused = self.polar_fusion(y_restored, H, W)  # (B, C, L)
        
        return y_fused

    def forward(self, x: torch.Tensor, **kwargs):
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


# ==============================================================================
# 4. VSSBlock
# ==============================================================================

class VSSBlock(nn.Module):
    def __init__(self, hidden_dim: int = 0, drop_path: float = 0, 
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6), 
                 attn_drop_rate: float = 0, d_state: int = 16, **kwargs):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SASS_SS2D(
            d_model=hidden_dim, 
            dropout=attn_drop_rate, 
            d_state=d_state,
            **kwargs
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


# ==============================================================================
# 5. VSSLayer (Encoder) - 带 FOV 位置编码
# ==============================================================================

class VSSLayer(nn.Module):
    """FOV 极坐标感知的 Encoder Layer"""
    def __init__(self, dim, depth, attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, 
                 downsample=None, use_checkpoint=False, d_state=16, **kwargs):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        
        # FOV 位置编码器
        self.fov_encoder = nn.Sequential(
            nn.Linear(2, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim)
        )
        
        # 缓存
        self._fov_cache = {}
        
        # VSSBlocks
        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim, 
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, 
                norm_layer=norm_layer, 
                attn_drop_rate=attn_drop, 
                d_state=d_state,
            )
            for i in range(depth)
        ])
        
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
    
    def get_fov_coords(self, H, W, device):
        """生成 FOV 极坐标（归一化）"""
        cache_key = (H, W, str(device))
        if cache_key in self._fov_cache:
            return self._fov_cache[cache_key]
            
        cy, cx = H / 2, W / 2
        y = torch.arange(H, device=device, dtype=torch.float32)
        x = torch.arange(W, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        dy = (yy - cy) / (H / 2)
        dx = (xx - cx) / (W / 2)
        
        r = torch.sqrt(dy**2 + dx**2).clamp(max=2.0)
        theta = torch.atan2(dy, dx) / math.pi
        
        coords = torch.stack([r, theta], dim=-1)
        self._fov_cache[cache_key] = coords
        return coords

    def forward(self, x):
        B, H, W, C = x.shape
        
        # 注入 FOV 位置信息
        fov_coords = self.get_fov_coords(H, W, x.device).unsqueeze(0)
        fov_embed = self.fov_encoder(fov_coords)
        x = x + fov_embed
        
        # VSSBlocks 处理
        for blk in self.blocks:
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        
        x_skip = x
        
        if self.downsample is not None:
            x = self.downsample(x)
            
        return x, x_skip


# ==============================================================================
# 6. VSSLayer_up (Decoder) - 暂不加 FOV 编码
# ==============================================================================

class VSSLayer_up(nn.Module):
    def __init__(self, dim, depth, attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, 
                 upsample=None, use_checkpoint=False, d_state=16, **kwargs):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        
        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim, 
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, 
                norm_layer=norm_layer, 
                attn_drop_rate=attn_drop, 
                d_state=d_state,
            )
            for i in range(depth)
        ])

    def forward(self, x, skip=None):
        if self.upsample is not None:
            x = self.upsample(x)
        
        if skip is not None:
            x = x + skip
            
        for blk in self.blocks:
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


# ==============================================================================
# 7. Dynamic FOV Gating (保持不变)
# ==============================================================================

class DynamicFOVGating(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_ch + 2, in_ch // 4, 3, 1, 1),
            nn.GroupNorm(8, in_ch // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 4, 1, 3, 1, 1),
            nn.Sigmoid() 
        )

    def get_polar_grid(self, H, W, device):
        y_coord, x_coord = torch.meshgrid(
            torch.arange(H, device=device), 
            torch.arange(W, device=device), 
            indexing='ij'
        )
        y_coord = y_coord.float()
        x_coord = x_coord.float()
        x_norm = 2 * (x_coord / (W - 1)) - 1
        y_norm = 2 * (y_coord / (H - 1)) - 1
        r = torch.sqrt(x_norm**2 + y_norm**2)
        theta = torch.atan2(y_norm, x_norm) / math.pi
        return torch.stack([r, theta], dim=0).unsqueeze(0)

    def forward(self, x):
        B, C, H, W = x.shape
        polar_grid = self.get_polar_grid(H, W, x.device).repeat(B, 1, 1, 1)
        r_map = polar_grid[:, 0:1, :, :]
        x_cat = torch.cat([x, polar_grid], dim=1) 
        soft_mask = self.gate_conv(x_cat)
        hard_mask = (r_map <= 1.15).float()
        final_mask = soft_mask * hard_mask
        return x * final_mask


# ==============================================================================
# 8. 主模型: VSSM_Polar
# ==============================================================================

class VSSM_Polar(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1, 
                 depths=[2, 2, 9, 2], depths_decoder=[2, 9, 2, 2],
                 dims=[96, 192, 384, 768], dims_decoder=[768, 384, 192, 96], 
                 d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True, use_checkpoint=False, 
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed = PatchEmbed2D(
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        # Encoder Layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        # Bottleneck
        self.fov_gate = DynamicFOVGating(dims[-1])

        # Decoder Layers
        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer_up(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if (i_layer != 0) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers_up.append(layer)

        self.final_up = Final_PatchExpand2D(dim=dims_decoder[-1], dim_scale=4, norm_layer=norm_layer)
        self.head_seg = nn.Conv2d(dims_decoder[-1] // 4, num_classes, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        skip_list = []
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        
        for layer in self.layers:
            x, x_skip = layer(x)
            skip_list.append(x_skip)
            
        return x, skip_list

    def forward_features_up(self, x, skip_list):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x, skip=None)
            else:
                skip_idx = -(inx + 1)
                x = layer_up(x, skip=skip_list[skip_idx])
        return x

    def forward(self, x):
        # 1. Encoder
        x, skip_list = self.forward_features(x)
        
        # 2. Bottleneck: FOV Gating
        x = x.permute(0, 3, 1, 2) 
        x = self.fov_gate(x)
        x = x.permute(0, 2, 3, 1) 
        
        # 3. Decoder
        x = self.forward_features_up(x, skip_list)
        
        # 4. Final Head
        x = self.final_up(x)
        x = x.permute(0, 3, 1, 2)
        out_seg = self.head_seg(x)
        
        return out_seg


# ==============================================================================
# 9. 外部调用接口
# ==============================================================================

class PGMamba(nn.Module):
    """
    PG-Mamba: Polar-Guided Mamba for OCTA Vessel Segmentation
    
    Args:
        input_channels: 输入通道数 (默认 3: image + x_map + y_map)
        num_classes: 输出类别数
        depths: Encoder 各层深度
        depths_decoder: Decoder 各层深度
        drop_path_rate: DropPath 比率
    """
    def __init__(self, input_channels=3, num_classes=1, 
                 depths=[2, 2, 9, 2], depths_decoder=[2, 9, 2, 2], 
                 drop_path_rate=0.2, load_ckpt_path=None):
        super().__init__()
        self.num_classes = num_classes
        self.vmunet = VSSM_Polar(
            in_chans=input_channels, 
            num_classes=num_classes, 
            depths=depths, 
            depths_decoder=depths_decoder, 
            drop_path_rate=drop_path_rate,
        )
        
        if load_ckpt_path is not None:
            self.load_pretrained(load_ckpt_path)

    def load_pretrained(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 过滤不匹配的键
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() 
                          if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} parameters from {ckpt_path}")

    def forward(self, x):
        logits = self.vmunet(x)
        return logits


# 兼容旧代码的别名
VMUNet_Polar = PGMamba


# ==============================================================================
# 10. 测试代码
# ==============================================================================

if __name__ == "__main__":
    # 测试模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    model = PGMamba(
        input_channels=3,  # image + x_map + y_map
        num_classes=1,
        depths=[2, 2, 9, 2],
        depths_decoder=[2, 9, 2, 2],
        drop_path_rate=0.2
    ).to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    # 测试前向传播
    x = torch.randn(2, 3, 512, 512).to(device)
    
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        y = model(x)
    
    print(f"Output shape: {y.shape}")
    print("Forward pass successful!")
    
    # 打印扫描方向信息
    print("\n" + "="*50)
    print("PG-Mamba V2 扫描方向配置:")
    print("="*50)
    print("o1: 主对角线蛇形 (45° 血管)")
    print("o2: 副对角线蛇形 (135° 血管)")
    print("o3: 径向向外 (从中心辐射)")
    print("o4: 径向向内 (从边缘汇聚)")
    print("="*50)