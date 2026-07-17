import openpyxl
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import to_pil_image, resize
import math
import ipdb
import torch.nn.functional as F
from torch.backends.cuda import sdp_kernel
from PIL import Image
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import gc
import os
import torch.nn as nn
from typing import List, Tuple
import numpy as np
import torch.nn.functional as F
from PIL import Image


# 根据输入规模动态计算注意力平移系数 beta。
# 参数 x 通常表示窗口数或参与计算的 token 规模，既可以是标量，也可以是
# 能被 NumPy 广播的数组；alpha 控制 beta 随规模增长的速度。
# 返回值为 1 + alpha * log(1 + x)，使用对数增长以避免窗口数增大时系数过快膨胀。
def dynamic_beta(x, alpha=0.2):
    return 1 + alpha * np.log1p(x)

# 根据高置信 token 的数量动态计算注意力缩放系数 gamma。
# 参数 x 是每个 query 对应的有效 token 数，alpha 决定反比修正项的强度；
# x 可以是标量或张量，返回值会保持相应的可广播形状。
# token 越少，返回的 gamma 越大，从而更强地放大稀疏的有效响应；调用方需保证 x 非零。
def dynamic_gamma(x, alpha=1):
    return 1  +  alpha / (x)

palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
        [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
        [107, 142, 35], [152, 251, 152], [70, 130, 180],
        [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
        [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]]

# 将二维语义类别索引图转换为便于可视化的 RGB 彩色图。
# 参数 img 的形状应为 [H, W]，每个像素保存类别编号；palette 中第 cls 项
# 就是该类别使用的 RGB 颜色。返回 uint8 类型的 [H, W, 3] NumPy 数组；
# 超出当前 palette 范围的类别不会被循环命中，因此保持初始化时的黑色。
def draw(img):
    ret = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for cls, color in enumerate(palette):
        ret[img == cls] = color
    return ret


class UnNormalize(object):
    # 创建“反归一化”变换，并将逐通道均值和标准差保存为 [1, C, 1, 1] 张量。
    # mean 和 std 应按输入图像的通道顺序提供。这里保存的是普通属性而非
    # nn.Module buffer，因此实际调用时会根据输入 image 的设备显式搬运。
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(1, -1, 1, 1)  # [1, C, 1, 1]
        self.std = torch.tensor(std).view(1, -1, 1, 1)

    # 把归一化后的图像恢复到原始数值空间，计算公式为 image * std + mean。
    # 支持单张 [C, H, W] 和批量 [B, C, H, W] 张量，并为两种输入构造
    # 可广播的统计量；其他维度会抛出 ValueError。返回新张量，不会原地修改 image。
    def __call__(self, image):
        if image.dim() == 3:  # [C, H, W]
            mean = self.mean.view(-1, 1, 1).to(image.device)
            std = self.std.view(-1, 1, 1).to(image.device)
        elif image.dim() == 4:  # [B, C, H, W]
            mean = self.mean.to(image.device)
            std = self.std.to(image.device)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
        
        return image * std + mean


# 通过迭代相似度聚合，为每个 token 构造语义 proxy 表示。
# ex_feats_grid 预期为 [1, N, C] 的 token 特征；model_cfg 提供迭代次数
# mini_iters 和相似度阈值 initial_crit_pos；scale 用于下一轮相似度计算。
# 每轮会聚合超过阈值的 token，并在某个 proxy 没有候选时恢复其自身连接，
# 防止该 proxy 消失。返回最终 proxy [1, N, C] 及最后一轮的二值关联矩阵
# mask_one [1, N, N]。num_heads 和 indices 当前为接口兼容参数，函数内未使用。
def proxy_sim(ex_feats_grid, model_cfg, num_heads=12, scale=1, indices=None):
    sim = torch.bmm(ex_feats_grid, ex_feats_grid.transpose(1, 2))  # [1, N, N]
    
    for mi in range(model_cfg.mini_iters):
        mask_one = (sim > model_cfg.initial_crit_pos) # [1, N, N]
        mask_one = mask_one.to(ex_feats_grid.dtype) 
        gone_proxy_idx = torch.where(mask_one.sum(dim=-1)[0] == 0)[0]              

        if len(gone_proxy_idx) != 0:
            print(f"Warning: {len(gone_proxy_idx)} proxies are gone in mini_iter {mi}. Restoring them.")
            mask_one[0, gone_proxy_idx, gone_proxy_idx] = 1 
        
        sum_proxy = torch.bmm(mask_one, ex_feats_grid)   # [1, N, C]
        count_proxy = mask_one.sum(dim=-1, keepdim=True) # [1, N, 1]
        proxy = sum_proxy / (count_proxy + 1e-6)
        proxy = F.normalize(proxy, dim=-1)
        sim = torch.bmm(proxy * scale, ex_feats_grid.transpose(1, 2))  # [1, N, N]
    
    return proxy, mask_one


class KV_Extension(nn.Module):
    # 初始化 ProxyCLIP 使用的跨窗口 Key/Value 扩展模块。
    # 模块本身没有可训练参数；cossim 保留为余弦相似度工具，当前 forward
    # 的主体计算通过归一化特征和批量矩阵乘法完成。
    def __init__(self):
        super().__init__()
        self.cossim = nn.CosineSimilarity(dim=-1, eps=1e-6)
    
    # 在所有滑动窗口的 token 上建立全局注意力，并输出每个窗口的增强特征。
    # ex_feats_grid 形状为 [h_grids, w_grids, C, H, W]，会被展平为
    # B*S 个全局 Key，其中 B=h_grids*w_grids、S=H*W；v_ext 是 CLIP 最后一层
    # Value，先插值到相同的 H*W 网格，再按多头形式整理为 [num_heads, B*S, Dh]。
    # indices 在 smoothing 开启时标识 token 对应的原图位置，用于联合同位置 token
    # 和窗口内八邻域生成 proxy；beta、gamma 形参为兼容参数，实际归一化配置主要
    # 从 model_cfg 读取。model_cfg 还控制 proxy 迭代、动态 beta/gamma、截断阈值、
    # temperature 等行为。返回 [S, B, C_out]，供 CLIP 最后一层的输出投影继续处理。
    # 注意：所有窗口 token 始终保留在 B*S 的 K/V 库中，本方法不会去除重叠 token。
    def forward(self, ex_feats_grid, num_heads=12, scale=1,
                 beta=1.2, gamma=3.0, indices=None, v_ext=None, model_cfg=None):

        h_grids, w_grids, C, H, W = ex_feats_grid.shape
        B = h_grids * w_grids
        S = H * W

        ex_feats_grid = ex_feats_grid.permute(0, 1, 3, 4, 2).reshape(B, S, C)   
        ex_feats_grid = ex_feats_grid.reshape(1, B * S, C)                           
        ex_feats_grid = F.normalize(ex_feats_grid, dim=-1)                   
        mask_one = None          
        attn_output = []
        
        v_ext = v_ext.reshape(B*num_heads, model_cfg.token_size[0], model_cfg.token_size[1], -1) \
                    .permute(0, 3, 1, 2).contiguous()                                       # [B*H, Dh, Th, Tw]
        v_ext = F.interpolate(v_ext, size=(H, W), mode='bilinear', align_corners=False)  # [B*H, Dh, H, W]
        
        Dh = v_ext.shape[1]
        v_ext = v_ext.view(B, num_heads, Dh, H, W).flatten(3)                    # [B,H,Dh,S]
        v_ext = v_ext.permute(1, 0, 3, 2).reshape(num_heads, B * S, Dh)          # [H,B*S,Dh]

        key = ex_feats_grid   
        if getattr(model_cfg, 'proxy_sim', False) or getattr(model_cfg, 'dynamic_gamma', False):
            if getattr(model_cfg, 'smoothing', False):
                indices = indices.flatten()
                indices_mask = indices.unsqueeze(0) == indices.unsqueeze(1)  # [N, N]
                device = key.device
                r = torch.arange(H, device=device)
                c = torch.arange(W, device=device)
                rr, cc = torch.meshgrid(r, c, indexing='ij')  # [H, W], [H, W]
                coords = torch.stack([rr.flatten(), cc.flatten()], dim=1)  # [H*W, 2]
                
                delta = coords.unsqueeze(1) - coords.unsqueeze(0)  # [H*W, H*W, 2]
                window_mask = (delta.abs().max(dim=-1)[0] <= 1) & (delta.abs().sum(dim=-1) > 0)
                neighbor_mask = torch.block_diag(*([window_mask] * (h_grids * w_grids))).to(device)  # [N, N]
                
                cand_mask = indices_mask | neighbor_mask   # [1, N, N]
                cand_mask.fill_diagonal_(False)
                
                count = cand_mask.sum(dim=-1, keepdim=True).clamp(min=1)
                avg_feats = (cand_mask.half() @ key) / count
                
                proxy = avg_feats
            else:
                proxy = key.clone()

            attn_weights = torch.bmm(F.normalize(proxy, dim=-1), key.transpose(1, 2))  # [1, N, N]
            
            if model_cfg.mini_iters == 0:
                mask_one = (attn_weights > model_cfg.initial_crit_pos) # [1, B*S, B*S]
                mask_one = mask_one.to(key.dtype) 
            else:
                for mi in range(model_cfg.mini_iters):
                    mask_one = (attn_weights > model_cfg.initial_crit_pos) # [1, B*S, B*S]
                    mask_one = mask_one.to(key.dtype) 
                    gone_proxy_idx = torch.where(mask_one.sum(dim=-1)[0] == 0)[0]              

                    if len(gone_proxy_idx) != 0:
                        mask_one[0, gone_proxy_idx, gone_proxy_idx] = 1 
                    
                    proxy = torch.bmm(mask_one, key)   # [1, N, C]
                    count_proxy = mask_one.sum(dim=-1, keepdim=True) # [1, N, 1]
                    proxy = proxy / (count_proxy + 1e-6)
                    proxy = F.normalize(proxy, dim=-1)
                    attn_weights = torch.bmm(proxy, key.transpose(1, 2))  # [1, N, N]
            
            if getattr(model_cfg, 'proxy_sim', False) == False:
                attn_weights = torch.bmm(key, key.transpose(1, 2))  # [1, N, N]
            
        else:
            attn_weights = torch.bmm(key, key.transpose(1, 2))  # [1, B*S, B*S]

        attn_weights = attn_weights.reshape(B, S, B*S) * scale

        if getattr(model_cfg, 'token_norm', False):
            beta_ = model_cfg.beta
            gamma_ = model_cfg.gamma
            sim_mean = attn_weights.mean(dim=-1, keepdim=True)            # [B,S,1]

            is_mask_cnt_exist = False
            if getattr(model_cfg, 'dynamic_gamma', False):
                if isinstance(mask_one, torch.Tensor):
                    mask_cnt = mask_one.sum(dim=-1).reshape(B, S, 1)      # [B,S,1]
                    is_mask_cnt_exist = True
                    gamma_ = dynamic_gamma(x=mask_cnt, alpha=model_cfg.gamma_alpha)
                else:
                    gamma_ = model_cfg.gamma

            if getattr(model_cfg, 'dynamic_beta', False):                
                
                if is_mask_cnt_exist == False:
                    mask_cnt = mask_one.sum(dim=-1).reshape(B, S, 1) 
                beta_ = dynamic_beta(x=B, alpha=model_cfg.beta_alpha)
                
            sim_mean[sim_mean < 0] = 0
            attn_weights = (attn_weights - sim_mean * beta_) * gamma_
        else:
            attn_weights = (attn_weights - attn_weights.mean() * model_cfg.beta) * model_cfg.gamma
        
        max_per_row = attn_weights.max(dim=-1, keepdim=True).values       # [B,S,1]
        cutting_hp = torch.as_tensor(getattr(model_cfg, 'cutting_hp', 0.0),
                                        device=attn_weights.device, dtype=attn_weights.dtype)
        dynamic_cutting_hp = torch.minimum(max_per_row, cutting_hp)       # [B,S,1]
        attn_weights.masked_fill_(attn_weights < dynamic_cutting_hp, float('-inf'))
        
        attn_weights = F.softmax(attn_weights / model_cfg.temperature, dim=-1)  # [B,S,B*S]

        if torch.isnan(attn_weights).any():
            attn_weights = torch.nan_to_num(attn_weights, neginf=float('-inf'))

        attn_output = torch.einsum('bsn,hnd->hsbd', attn_weights, v_ext)  # [H, S, B, Dh]
        attn_output = attn_output.permute(1, 2, 0, 3).reshape(S, B, -1)

        return attn_output  # [num_tokens=S, B, C]


class KV_Extension_ClearCLIP(nn.Module):
    # 初始化 ClearCLIP 分支使用的跨窗口 Key/Value 扩展模块。
    # 与 ProxyCLIP 版本一样，该模块没有可训练参数；cossim 当前仅作为预留工具，
    # 真正的相关性由 forward 中的归一化点积得到。
    def __init__(self):
        super().__init__()
        self.cossim = nn.CosineSimilarity(dim=-1, eps=1e-6)
    
    # 在 ClearCLIP 的 query 特征和 value 特征上执行跨窗口全局注意力。
    # ex_feats_grid 形状为 [h_grids, w_grids, S, C]，其中 S 应能还原为方形
    # H*W token 网格；v_ext 会被插值到该网格，并整理成 [num_heads, B*S, Dh]。
    # 当 proxy_sim 或 dynamic_gamma 开启时，indices 用于寻找跨窗口的同位置 token，
    # 同时结合窗口内八邻域计算 proxy，之后按 model_cfg 中的阈值、动态归一化参数
    # 和 temperature 生成全局注意力。lbl_grid、beta、gamma 当前为接口兼容参数，
    # 未直接参与主体计算。返回形状为 [S, B, C_out] 的跨窗口聚合结果。
    # 本实现同样保留全部 B*S 个 K/V 条目，不会对重叠窗口 token 做唯一化处理。
    def forward(self, ex_feats_grid, num_heads=12, scale=1, lbl_grid=None,
                 beta=1.2, gamma=3.0, indices=None, v_ext=None, model_cfg=None):

        h_grids, w_grids, S, C  = ex_feats_grid.shape
        B = h_grids * w_grids
        H = W = int(S ** 0.5)
        
        ex_feats_grid = ex_feats_grid.reshape(B, S, C)   
        ex_feats_grid = ex_feats_grid.reshape(1, B * S, C)            
            
        mask_one = None        
        attn_output = []

        v_ext = v_ext.reshape(B*num_heads, model_cfg.token_size[0], model_cfg.token_size[1], -1) \
                    .permute(0, 3, 1, 2).contiguous()                                       
        v_ext = F.interpolate(v_ext, size=(H, W), mode='bilinear', align_corners=False)  
        Dh = v_ext.shape[1]
        
        v_ext = v_ext.view(B, num_heads, Dh, H, W).flatten(3)                    # [B,H,Dh,S]
        v_ext = v_ext.permute(1, 0, 3, 2).reshape(num_heads, B * S, Dh)          # [H,B*S,Dh]

        if getattr(model_cfg, 'proxy_sim', False) or getattr(model_cfg, 'dynamic_gamma', False):
            # 1) indices mask
            indices = indices.flatten()
            indices_mask = indices.unsqueeze(0) == indices.unsqueeze(1)  # [N, N]
            
            # 2) Neighbor mask
            device = ex_feats_grid.device
            r = torch.arange(H, device=device)
            c = torch.arange(W, device=device)
            rr, cc = torch.meshgrid(r, c, indexing='ij')  # [H, W], [H, W]
            coords = torch.stack([rr.flatten(), cc.flatten()], dim=1)  # [H*W, 2]
            
            delta = coords.unsqueeze(1) - coords.unsqueeze(0)  # [H*W, H*W, 2]
            window_mask = (delta.abs().max(dim=-1)[0] <= 1) & (delta.abs().sum(dim=-1) > 0)  

            neighbor_mask = torch.block_diag(*([window_mask] * (h_grids * w_grids))).to(device)  # [N, N]            
            cand_mask = indices_mask | neighbor_mask   # [1, N, N]
            
            count = cand_mask.sum(dim=-1, keepdim=True).clamp(min=1)
            avg_feats = (cand_mask.half() @ ex_feats_grid) / count
            proxy = avg_feats

            ex_feats_grid = F.normalize(ex_feats_grid, dim=-1)
            key = ex_feats_grid
            attn_weights = torch.bmm(F.normalize(proxy, dim=-1), key.transpose(1, 2))  # [1, N, N]
            
            if model_cfg.mini_iters == 0:
                mask_one = (attn_weights > model_cfg.initial_crit_pos) # [1, B*S, B*S]
                mask_one = mask_one.to(key.dtype) 
            else:
                for mi in range(model_cfg.mini_iters):
                    mask_one = (attn_weights > model_cfg.initial_crit_pos) # [1, B*S, B*S]
                    mask_one = mask_one.to(key.dtype) 
                    gone_proxy_idx = torch.where(mask_one.sum(dim=-1)[0] == 0)[0]              

                    if len(gone_proxy_idx) != 0:
                        mask_one[0, gone_proxy_idx, gone_proxy_idx] = 1 
                    
                    proxy = torch.bmm(mask_one, key)   # [1, N, C]
                    count_proxy = mask_one.sum(dim=-1, keepdim=True) # [1, N, 1]
                    proxy = proxy / (count_proxy + 1e-6)
                    proxy = F.normalize(proxy, dim=-1)
                    attn_weights = torch.bmm(proxy, key.transpose(1, 2))  # [1, N, N]
        else:
            ex_feats_grid = F.normalize(ex_feats_grid, dim=-1)
            key = ex_feats_grid
            attn_weights = torch.bmm(key, key.transpose(1, 2))

        attn_weights = attn_weights.reshape(B, S, B*S) * scale

        if getattr(model_cfg, 'token_norm', False):
            beta_ = model_cfg.beta
            gamma_ = model_cfg.gamma
            sim_mean = attn_weights.mean(dim=-1, keepdim=True)            # [B,S,1]

            if getattr(model_cfg, 'dynamic_gamma', False):
                if isinstance(mask_one, torch.Tensor):
                    mask_cnt = mask_one.sum(dim=-1).reshape(B, S, 1)      # [B,S,1]
                    gamma_ = dynamic_gamma(x=mask_cnt, alpha=model_cfg.gamma_alpha)
                else:
                    gamma_ = model_cfg.gamma

            if getattr(model_cfg, 'dynamic_beta', False):                
                beta_ = dynamic_beta(total_token=B*S, high_conf_token=mask_cnt, alpha=model_cfg.beta_alpha)                

            sim_mean[sim_mean < 0] = 0   
            attn_weights = (attn_weights - sim_mean * beta_) * gamma_
        else:
            attn_weights = (attn_weights - attn_weights.mean() * model_cfg.beta) * model_cfg.gamma
        
        max_per_row = attn_weights.max(dim=-1, keepdim=True).values       # [B,S,1]
        cutting_hp = torch.as_tensor(getattr(model_cfg, 'cutting_hp', 0.0),
                                        device=attn_weights.device, dtype=attn_weights.dtype)
        dynamic_cutting_hp = torch.minimum(max_per_row, cutting_hp)       # [B,S,1]
        attn_weights.masked_fill_(attn_weights < dynamic_cutting_hp, float('-inf'))

        attn_weights = F.softmax(attn_weights / model_cfg.temperature, dim=-1)  # [B,S,B*S]

        if torch.isnan(attn_weights).any():
            attn_weights = torch.nan_to_num(attn_weights, neginf=float('-inf'))

        attn_output = torch.einsum('bsn,hnd->hsbd', attn_weights, v_ext)  # [H, S, B, Dh]
        attn_output = attn_output.permute(1, 2, 0, 3).reshape(S, B, -1)

        return attn_output  


