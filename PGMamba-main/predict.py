import torch
import os
import sys
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.morphology import skeletonize

# ==========================================
# 1. Custom Module Import
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from modelszoo import VMUNet_Polar
    from dataset import SegmentationDataset
    
    # [CRITICAL CHANGE] Import calc_result from evaluation.py
    # This ensures we use the exact same Normalize + Otsu logic as training
    from evaluation import calc_result 
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please ensure dataset.py, evaluation.py, and modelszoo are in the current directory.")
    sys.exit(1)

# ==========================================
# 2. Configuration
# ==========================================
class Config:
    # Update these paths if necessary
    MODEL_PATH = '/root/autodl-tmp/OCTAMamba/result/VMUNet_Polar/OCTA_Custom/2026-01-25_18-24-27_Run/model_best.pth'
    TEST_DATA_PATH = '/root/autodl-tmp/OCTAMamba/data/test' 
    OUTPUT_FOLDER = '/root/autodl-tmp/OCTAMamba/predictions/VMUNet_Polar2/OCTA_Custom'
    
    TILE_SIZE = 512
    NUM_CLASSES = 1
    INPUT_CHANNELS = 3 
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# 3. Supplemental Metric Functions
# ==========================================
def calculate_extra_metrics(pred_prob, gt_mask):
    """
    Calculates metrics NOT present in evaluation.py (like clDice, MAE, SoftDice)
    to complement the standard metrics.
    """
    smooth = 1e-5
    
    # Use Otsu thresholding locally to match the logic of evaluation.py for consistency
    pred_u8 = (pred_prob * 255).astype(np.uint8)
    otsu_thresh, _ = cv2.threshold(pred_u8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pred_bin = (pred_u8 > otsu_thresh).astype(np.float32)
    gt_bin = (gt_mask > 0.5).astype(np.float32)

    # 1. Soft Dice
    inter_soft = (pred_prob * gt_bin).sum()
    union_soft = (pred_prob).sum() + (gt_bin).sum()
    soft_dice = (2 * inter_soft + smooth) / (union_soft + smooth)

    # 2. MAE
    mae = np.mean(np.abs(pred_prob - gt_bin))

    # 3. clDice (Centerline Dice)
    try:
        if pred_bin.sum() > 0 and gt_bin.sum() > 0:
            skel_pred = skeletonize(pred_bin.astype(np.uint8))
            skel_gt = skeletonize(gt_bin.astype(np.uint8))
            
            t_prec = (skel_pred * gt_bin).sum() / (skel_pred.sum() + smooth)
            t_sens = (skel_gt * pred_bin).sum() / (skel_gt.sum() + smooth)
            
            cldice = 2 * t_prec * t_sens / (t_prec + t_sens + smooth)
        else:
            cldice = 0.0
    except Exception:
        cldice = 0.0

    return {
        "SoftDice": soft_dice,
        "clDice": cldice,
        "MAE": mae
    }

# ==========================================
# 4. Main Inference Loop
# ==========================================
def main():
    print("="*80)
    print(f"OCTAMamba Aligned Evaluation (Training Logic) | Device: {Config.DEVICE}")
    print("="*80)

    # Setup directories
    save_continuous = os.path.join(Config.OUTPUT_FOLDER, 'continuous')
    save_binary = os.path.join(Config.OUTPUT_FOLDER, 'binary')
    os.makedirs(save_continuous, exist_ok=True)
    os.makedirs(save_binary, exist_ok=True)

    # Initialize DataLoader
    print("[1/4] Initializing DataLoader (Mode: Test)...")
    try:
        test_dataset = SegmentationDataset(
            ls_path_dataset=Config.TEST_DATA_PATH,
            image_dir_name="images", 
            label_dir_name="labels", 
            label_prefix="mask_", 
            mode="test", crop_size=Config.TILE_SIZE
        )
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    except Exception as e:
        print(f"❌ Dataset Error: {e}")
        return

    # Load Model
    print("\n[2/4] Loading Model...")
    model = VMUNet_Polar.VMUNet_Polar(input_channels=Config.INPUT_CHANNELS, num_classes=Config.NUM_CLASSES)
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
    model.to(Config.DEVICE)
    model.eval()

    # Inference Loop
    print("\n[3/4] Running Inference & Evaluation...")
    all_metrics = []
    
    with torch.no_grad():
        for batch_idx, batch_data in tqdm(enumerate(test_loader), total=len(test_loader)):
            name = batch_data[0][0]
            data = batch_data[1].to(Config.DEVICE)
            label = batch_data[2].to(Config.DEVICE)
            
            # Get original size
            original_size = None
            if len(batch_data) > 3:
                size_tensor = batch_data[3]
                original_size = (size_tensor[0][0].item(), size_tensor[0][1].item())

            # Forward pass
            logits = model(data)
            pred_prob_tensor = torch.sigmoid(logits)

            # Convert to Numpy
            pred_prob = pred_prob_tensor[0][0].cpu().numpy()
            gt_mask = label[0][0].cpu().numpy()

            # Crop Padding
            if original_size is not None:
                h, w = original_size
                pred_prob = pred_prob[:h, :w]
                gt_mask = gt_mask[:h, :w]

            # Save Images
            # 1. Continuous Probability Map
            cv2.imwrite(os.path.join(save_continuous, name), (pred_prob * 255).astype(np.uint8))
            
            # 2. Binary Map (Visualizing using Otsu to match metric logic)
            pred_u8 = (pred_prob * 255).astype(np.uint8)
            otsu_thresh, _ = cv2.threshold(pred_u8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            pred_bin_vis = (pred_u8 > otsu_thresh).astype(np.uint8) * 255
            cv2.imwrite(os.path.join(save_binary, name), pred_bin_vis)

            # --- METRIC CALCULATION (ALIGNED) ---
            
            # 1. Standard Metrics (Dice, IoU, Acc, Sen, Spe) 
            # using evaluation.py logic (Normalize + Otsu)
            # This should reproduce your ~95% Dice
            standard_metrics = calc_result(pred_prob, gt_mask)
            
            # 2. Extra Metrics (clDice, SoftDice, MAE)
            extra_metrics = calculate_extra_metrics(pred_prob, gt_mask)
            
            # Merge metrics
            metrics = {**standard_metrics, **extra_metrics}
            metrics['Filename'] = name
            
            all_metrics.append(metrics)

    # 5. Generate Report
    if all_metrics:
        print("\n[4/4] Generating Report")
        df = pd.DataFrame(all_metrics)
        
        # Save Detailed CSV
        csv_path = os.path.join(Config.OUTPUT_FOLDER, 'detailed_metrics.csv')
        # Reorder columns for better readability
        cols_order = ['Filename', 'dice', 'iou', 'clDice', 'SoftDice', 'MAE', 'sen', 'spe', 'acc']
        # Handle case sensitivity if keys differ in calc_result
        available_cols = [c for c in cols_order if c in df.columns] 
        df = df[['Filename'] + [c for c in df.columns if c != 'Filename']] # Fallback sort
        
        df.to_csv(csv_path, index=False)
        print(f"✓ Detailed metrics saved: {csv_path}")

        # Calculate Statistics
        mean_series = df.mean(numeric_only=True)
        std_series = df.std(numeric_only=True)
        
        summary_df = pd.DataFrame({
            'Metric': mean_series.index,
            'Mean': mean_series.values,
            'Std Dev': std_series.values
        })

        # Save Summary CSV
        summary_csv_path = os.path.join(Config.OUTPUT_FOLDER, 'summary_metrics.csv')
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"✓ Summary metrics saved: {summary_csv_path}")

        # Generate Visual Report (.txt)
        report_lines = []
        report_lines.append("="*55)
        report_lines.append(f"{'FINAL PERFORMANCE SUMMARY (TRAINING ALIGNED)':^55}")
        report_lines.append("="*55)
        report_lines.append(f"{'Metric':<15} | {'Mean':^15} | {'Std Dev':^15}")
        report_lines.append("-" * 55)
        
        for _, row in summary_df.iterrows():
            metric = row['Metric']
            mean_val = row['Mean']
            std_val = row['Std Dev']
            
            if metric == 'MAE' or metric == 'loss':
                line = f"{metric:<15} | {mean_val:^15.4f} | {std_val:^15.4f}"
            else:
                line = f"{metric:<15} | {mean_val*100:^14.2f}% | {std_val*100:^14.2f}%"
            
            report_lines.append(line)
        
        report_lines.append("-" * 55)
        report_lines.append(f"Samples: {len(df)}")
        report_lines.append("Note: Logic strictly aligned with evaluation.py (Otsu)")
        report_lines.append("="*55)

        report_text = "\n".join(report_lines)
        print("\n" + report_text)

        txt_path = os.path.join(Config.OUTPUT_FOLDER, 'summary_report.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"✓ Visual report saved: {txt_path}")

if __name__ == "__main__":
    main()