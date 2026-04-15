import numpy as np
import math
import cv2

def calc_result(np_pred: np.ndarray, np_label: np.ndarray, thresh_value=None):
    # 1. 预处理:归一化并转为 uint8 以适配 OpenCV
    # 注意:输入可能是 Logits 或 Probability,这里统一归一化到 0-255
    if np_pred.max() != np_pred.min(): # 防止全黑图导致除零
        temp = cv2.normalize(np_pred, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    else:
        temp = np.zeros_like(np_pred, dtype="uint8")

    # 2. 阈值处理 (二值化)
    if thresh_value is None:
        # 使用 Otsu 自动阈值
        _, np_pred_bin = cv2.threshold(temp, 0.0, 1.0, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # 使用固定阈值 (假设输入是 0-1 的概率,这里需要对应转换或者直接用 0-255)
        # 你的原代码逻辑是将 temp (0-255) 与 thresh_value 比较。
        # 如果 thresh_value 是 0.5 (概率),这里会有问题。
        # 通常我们建议:如果使用了 normalize 到 255,阈值应该是 127。
        # 为了兼容你原有的调用习惯,这里保持原逻辑,但建议在外面尽量用 None (Otsu)
        _, np_pred_bin = cv2.threshold(temp, thresh_value, 1.0, cv2.THRESH_BINARY)

    # 3. 展平
    np_pred = np_pred_bin.flatten()
    np_label = np_label.flatten()

    # ❌ [已删除] 导致崩溃的 Assert 语句
    # uni = np.unique(np_label)
    # assert (len(uni)==2) and (1 in uni) and (0 in uni)

    # 4. 确保标签也是 0/1 (防止标签是 0/255)
    if np_label.max() > 1:
        np_label = (np_label > 0).astype(float)

    # 5. 计算混淆矩阵
    FP = np.sum(np.logical_and(np_pred == 1, np_label == 0)).astype(float)
    FN = np.sum(np.logical_and(np_pred == 0, np_label == 1)).astype(float)
    TP = np.sum(np.logical_and(np_pred == 1, np_label == 1)).astype(float)
    TN = np.sum(np.logical_and(np_pred == 0, np_label == 0)).astype(float)

    # 6. 计算指标 (使用 smooth 防止除以零)
    result = {}
    smooth = 1e-6 # 稍微加大一点 smooth 避免精度问题

    result['acc'] = (TP + TN) / (FP + FN + TP + TN + smooth)
    result['fdr'] = (FP + smooth)  / (FP + TP + smooth)
    
    sen = (TP + smooth) / (FN + TP + smooth)
    spe = (TN + smooth) / (FP + TN + smooth)
    
    result['sen'] = sen
    result['spe'] = spe
    result['gmean'] = math.sqrt(sen * spe)
    
    result['iou'] = (TP + smooth) / (FP + FN + TP + smooth)
    result['dice'] = (2.0 * TP + smooth) / (FP + FN + 2.0 * TP + smooth)
    
    # 额外计算 loss 用于打印 (模拟值,因为二值化后无法计算交叉熵)
    # 你的 run_benchmark.py 需要 'loss' 键来记录
    result['loss'] = 1.0 - result['dice'] 
    
    return result

def avg_result(ls_result):
    total_result = {}
    if not ls_result:
        return {}
        
    for r in ls_result:
        for key in r:
            if key not in total_result:
                total_result[key] = []
            total_result[key].append(r[key])
            
    for key in total_result:
        values = np.array(total_result[key])
        total_result[key] = float(values.mean())
        
    return total_result