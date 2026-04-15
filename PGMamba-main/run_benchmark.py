import os
import sys
import json
import time
import random
import shutil
import numpy as np
from datetime import datetime
from tqdm import tqdm
from loss import CombinedLoss
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

# ==========================================
# 自定义模块导入 (确保文件在同一目录下或PYTHONPATH中)
# ==========================================
try:
    from dataset import prepareDatasets
    from utils import traverseDataset
    from loss import DiceLoss
    from settings_benchmark import models
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    sys.exit(1)


# ==========================================
# 全局配置 (Configuration)
# ==========================================
class Config:
    ROOT_RESULT = "result"          # 结果保存根目录
    BATCH_SIZE = 4                  # 训练 Batch Size
    NUM_WORKERS = 4                 # Dataloader Workers
    LR_INIT = 0.0003                # 初始学习率
    WEIGHT_DECAY = 1e-2             # 权重衰减
    MAX_EPOCH = 400                 # 最大 Epoch
    EARLY_STOP_PATIENCE = 50        # 早停耐心值 (Epochs)
    SEED = 0                        # 随机种子

# ==========================================
# 辅助工具函数
# ==========================================
class NumpyEncoder(json.JSONEncoder):
    """用于解决 JSON dump 无法序列化 numpy 类型的问题"""
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def set_seed(seed=0):
    """固定随机种子以保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"🌱 Random Seed set to: {seed}")

def select_device():
    """交互式选择显卡"""
    count = torch.cuda.device_count()
    if count == 0:
        print("❌ No GPU found. Using CPU.")
        return torch.device('cpu')
    
    if count == 1:
        return torch.device('cuda:0')

    while True:
        try:
            s = input(f"🖥️  Found {count} GPUs. Choose ID (0-{count-1}): ")
            id_card = int(s)
            if 0 <= id_card < count:
                return torch.device(f'cuda:{id_card}')
        except ValueError:
            pass
        print("Invalid input, please try again.")

# ==========================================
# 核心训练逻辑
# ==========================================
def run_training_pipeline():
    # 1. 初始化环境
    set_seed(Config.SEED)
    device = select_device()
    print(f"✅ Using Device: {device}")

    # 2. 准备数据和模型列表
    all_dataset = prepareDatasets()
    print(f"\n📋 Model List: {list(models.keys())}")
    print(f"📂 Dataset List: {list(all_dataset.keys())}")

    os.makedirs(Config.ROOT_RESULT, exist_ok=True)

    # 3. 双层循环：遍历模型 -> 遍历数据集
    for name_model in models:
        root_model = os.path.join(Config.ROOT_RESULT, name_model)
        os.makedirs(root_model, exist_ok=True)

        for name_dataset in all_dataset:
            dataset = all_dataset[name_dataset]
            root_dataset = os.path.join(root_model, name_dataset)
            os.makedirs(root_dataset, exist_ok=True)

            # --- 实验目录管理 (带时间戳) ---
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_dir = os.path.join(root_dataset, f"{timestamp}_Run")
            os.makedirs(run_dir, exist_ok=True)

            # 初始化 TensorBoard
            writer = SummaryWriter(log_dir=run_dir)

            print(f"\n{'='*60}")
            print(f"🚀 Start Training: [{name_model}] on [{name_dataset}]")
            print(f"📁 Log Directory: {run_dir}")
            print(f"{'='*60}")

            # --- 数据加载器 ---
            train_loader = DataLoader(dataset['train'], batch_size=Config.BATCH_SIZE, shuffle=True, 
                                      drop_last=False, num_workers=Config.NUM_WORKERS)
            val_loader = DataLoader(dataset['val'], batch_size=1, num_workers=1)
            test_loader = DataLoader(dataset['test'], batch_size=1, num_workers=1)

            # --- 模型初始化 ---
            print(f"⚙️  Initializing model...")
            model = models[name_model]().to(device)
               
            criterion = dataset.get('loss', CombinedLoss(dice_weight=0.5, mse_weight=0.5))
            
            thresh_value = dataset.get('thresh', None)

            # --- 优化器与调度器 ---
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=Config.LR_INIT, 
                weight_decay=Config.WEIGHT_DECAY
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=Config.MAX_EPOCH, eta_min=1e-6)

            # --- 训练状态追踪 ---
            best_metric = -1
            best_epoch = -1
            training_log = []

            # 保存配置信息
            with open(os.path.join(run_dir, "config.txt"), "w") as f:
                f.write(f"Model: {name_model}\nDataset: {name_dataset}\n")
                f.write(f"Config: {json.dumps(Config.__dict__, indent=2, default=str)}\n")
                f.write(f"Loss: {criterion.__class__.__name__}\n")

            # ================= Epoch 循环 =================
            for epoch in range(Config.MAX_EPOCH):
                torch.cuda.empty_cache()
                current_lr = optimizer.param_groups[0]['lr']

                # 1. 训练阶段
                res_train = traverseDataset(
                    model=model, loader=train_loader, thresh_value=thresh_value,
                    log_section="Train", log_writer=None,
                    description=f"Ep {epoch}/{Config.MAX_EPOCH} (Train)",
                    device=device, funcLoss=criterion, optimizer=optimizer
                )
                
                # TensorBoard 记录
                writer.add_scalar('Train/Loss', res_train['loss'], epoch)
                writer.add_scalar('Train/LR', current_lr, epoch)

                # 2. 验证阶段
                res_val = traverseDataset(
                    model=model, loader=val_loader, thresh_value=thresh_value,
                    log_section="Val", log_writer=None,
                    description=f"Ep {epoch}/{Config.MAX_EPOCH} (Val)",
                    device=device, funcLoss=criterion, optimizer=None
                )

                # TensorBoard 记录
                writer.add_scalar('Val/Dice', res_val['dice'], epoch)
                writer.add_scalar('Val/IoU', res_val.get('iou', 0), epoch)
                writer.add_scalar('Val/Loss', res_val['loss'], epoch)

                # 3. 调度器步进
                scheduler.step()

                # 4. 保存最佳模型逻辑
                if res_val['dice'] > best_metric:
                    best_metric = res_val['dice']
                    best_epoch = epoch
                    
                    print(f"🔥 New Best! Epoch {epoch} | Dice: {best_metric:.4f}")
                    
                    # 保存权重
                    torch.save(model.state_dict(), os.path.join(run_dir, 'model_best.pth'))
                    
                    # 记录日志
                    log_entry = f"Epoch {epoch}: New Best Val Dice = {best_metric:.4f}"
                    training_log.append(log_entry)
                    
                    # 立即测试最佳模型
                    res_test = traverseDataset(
                        model=model, loader=test_loader, thresh_value=thresh_value,
                        log_section=None, log_writer=None,
                        description="Test Best", device=device, funcLoss=criterion
                    )
                    writer.add_scalar('Test/Best_Dice', res_test['dice'], epoch)
                    training_log.append(f"-> Test Result: Dice = {res_test['dice']:.4f}")
                    
                    # 实时写入 JSON (防止中途断电丢失)
                    with open(os.path.join(run_dir, "training_log.json"), "w") as f:
                        json.dump(training_log, f, indent=2, cls=NumpyEncoder)
                
                else:
                    print(f"   Epoch {epoch} | Dice: {res_val['dice']:.4f} (Best: {best_metric:.4f})")

                # 5. 早停机制 (Early Stopping)
                if epoch - best_epoch >= Config.EARLY_STOP_PATIENCE:
                    print(f"🛑 Early Stopping triggered! No improvement for {Config.EARLY_STOP_PATIENCE} epochs.")
                    break

            # 结束当前训练任务
            writer.close()
            with open(os.path.join(run_dir, "finished.flag"), "w") as f:
                f.write("Training Finished Successfully")
            
            print(f"✅ Finished: {name_model} on {name_dataset}")

if __name__ == "__main__":
    run_training_pipeline()