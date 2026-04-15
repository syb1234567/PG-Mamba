import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# 引入你的 evaluation 和 loss 模块
from evaluation import calc_result, avg_result
# 注意：不需要在这里 import loss，因为 funcLoss 是传进来的

def predict_sliding_window(model, image, tile_size=512, overlap=0.5, num_classes=1):
    """
    滑窗推理核心函数 (Sliding Window Inference)
    返回: 概率图 (Probabilities) [0, 1]
    """
    b, c, h, w = image.shape
    stride = int(tile_size * (1 - overlap))
    
    # 初始化累加器
    probs = torch.zeros((b, num_classes, h, w), device=image.device)
    counts = torch.zeros((b, num_classes, h, w), device=image.device)
    
    rows = list(range(0, h - tile_size + 1, stride))
    cols = list(range(0, w - tile_size + 1, stride))
    if rows[-1] + tile_size < h: rows.append(h - tile_size)
    if cols[-1] + tile_size < w: cols.append(w - tile_size)
    
    model.eval()
    with torch.no_grad():
        for y in rows:
            for x in cols:
                patch = image[:, :, y:y+tile_size, x:x+tile_size]
                logits = model(patch)
                prob = torch.sigmoid(logits) # Logits -> Probs
                
                probs[:, :, y:y+tile_size, x:x+tile_size] += prob
                counts[:, :, y:y+tile_size, x:x+tile_size] += 1
                
    final_prob = probs / counts
    return final_prob

def traverseDataset(model: nn.Module, loader: DataLoader,
                    description, device, funcLoss,
                    log_writer: SummaryWriter, log_section, optimizer=None,
                    show_result=False, thresh_value=None):
    is_training = (optimizer is not None)
    import time
    time_start = time.time()
    
    with tqdm(loader, unit="batch") as tepoch:
        total_loss = 0
        ls_eval_result = []
        model.train(is_training)
        
        for i, batch_data in enumerate(tepoch):
            # 解包数据
            name = batch_data[0]
            data = batch_data[1].to(device)
            label = batch_data[2].to(device)
            
            # 提取原始尺寸
            original_size = None
            if not is_training and len(batch_data) > 3:
                try:
                    size_tensor = batch_data[3] 
                    if size_tensor.dim() == 2:
                        org_h, org_w = size_tensor[0, 0].item(), size_tensor[0, 1].item()
                    else:
                        org_h, org_w = size_tensor[0].item(), size_tensor[1].item()
                    original_size = (org_h, org_w)
                except Exception:
                    original_size = None

            tepoch.set_description(description)
            eval_result = {}

            # ================= 训练阶段 =================
            if is_training:
                # 训练直接前向传播
                logits = model(data)
                
                if isinstance(logits, list):
                    logits = logits[0]
                
                # 计算 Loss (输入 Logits)
                loss = funcLoss(logits, label)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                loss_val = loss.item()

            # ================= 验证/测试阶段 =================
            else: 
                with torch.no_grad():
                    _, _, h, w = data.shape
                    
                    # 判断是否需要滑窗
                    if h > 512 or w > 512:
                        # [Path A] 滑窗推理 (返回的是概率图 Probs)
                        out = predict_sliding_window(model, data, tile_size=512, overlap=0.5)
                        loss_val = 0.0 
                    else:
                        # [Path B] 直接推理
                        logits = model(data)
                        
                        # 1. 先计算 Loss (输入 Logits)
                        loss = funcLoss(logits, label)
                        loss_val = loss.item()
                        
                        # 2. 再转为概率图 (用于后续指标计算)
                        out = torch.sigmoid(logits)
                    
                    # 后处理与评估
                    for index in range(loader.batch_size):
                        pred_np = out[index][0].detach().cpu().numpy()
                        gt_np = label[index][0].detach().cpu().numpy()
                        
                        # 裁剪回真实尺寸
                        if original_size is not None:
                            real_h, real_w = original_size
                            pred_np = pred_np[:real_h, :real_w]
                            gt_np = gt_np[:real_h, :real_w]
                        
                        # ====================================================
                        # 🔥 关键修改：强制使用固定阈值 0.5 (Fixed Threshold)
                        # ====================================================
                        # 我们不传概率图给 calc_result，而是传二值化后的图。
                        # 这样即使 evaluation.py 里有 Normalize+Otsu 逻辑，
                        # 因为输入已经是 0和1，Otsu 算出来的结果还是 0和1，
                        # 从而实现了“绕过 Otsu，强制 0.5”的效果。
                        pred_bin = (pred_np > 0.5).astype(np.float32)
                        
                        # 计算指标 (传入处理好的 pred_bin)
                        eval_result = calc_result(pred_bin, gt_np, thresh_value=None)

            # 记录 Loss
            eval_result["loss"] = float(loss_val)
            ls_eval_result.append(eval_result)

            total_loss += loss_val
            avg_loss = total_loss / (i + 1)
            
            # ====================================================
            # 🔥 关键修改：显示学习率 (LR)
            # ====================================================
            current_lr = optimizer.param_groups[0]['lr'] if optimizer else 0.0
            tepoch.set_postfix(
                lr=f"{current_lr:.2e}",  # 显示科学计数法
                loss=f"{avg_loss:.3f}", 
                curr=f"{loss_val:.3f}"
            )

    time_end = time.time()
    avg_ms = (time_end - time_start) * 1000 / len(loader) / max(loader.batch_size, 1)
    num_params = sum([param.nelement() for param in model.parameters()])

    result = avg_result(ls_eval_result)
    result['avg_ms'] = avg_ms
    result['num_params'] = num_params

    return result