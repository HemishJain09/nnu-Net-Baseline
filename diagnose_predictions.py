import argparse
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import pandas as pd
import random

def dice_score(pred, true):
    intersection = np.sum(pred[true == 1])
    return (2. * intersection) / (np.sum(pred) + np.sum(true) + 1e-8)

def diagnose_predictions(val_dir: Path, gt_dir: Path, marksheet_path: Path, sample_size: int = 50):
    print("="*60)
    print("🔍 DIAGNOSING PCNN RAW PREDICTIONS")
    print("="*60)
    
    df = pd.read_csv(marksheet_path)
    df['patient_id'] = df['patient_id'].astype(str)
    pcnn_patients = set(df[df['center'] == 'PCNN']['patient_id'].tolist())
    
    all_npz = sorted(list(val_dir.glob("*.npz")))
    pcnn_npz = [f for f in all_npz if f.name.split("_")[0] in pcnn_patients]
    
    if len(pcnn_npz) == 0:
        print("❌ No PCNN .npz files found!")
        return
        
    print(f"Found {len(pcnn_npz)} PCNN .npz files. Sampling {min(sample_size, len(pcnn_npz))} for diagnostics...")
    
    # Shuffle so we get a random sample of patients
    random.seed(42)
    sample_files = random.sample(pcnn_npz, min(sample_size, len(pcnn_npz)))
    
    stats = {
        "max_conf_under_10_percent": 0,
        "max_conf_under_1_percent": 0,
        "average_max_conf": [],
        "dice_at_10_percent": [],
        "dice_at_50_percent": [],
        "blank_predictions_at_10_percent": 0,
        "true_positives": 0
    }
    
    for npz_file in sample_files:
        case_id = npz_file.name.replace(".npz", "")
        gt_file = gt_dir / f"{case_id}.nii.gz"
        
        if not gt_file.exists():
            continue
            
        # Load raw probabilities
        data = np.load(npz_file)
        cancer_prob = np.array(data['probabilities'][1], dtype=np.float32)
        
        # Load ground truth
        gt_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_file)))
        has_cancer = np.sum(gt_arr) > 0
        if has_cancer:
            stats["true_positives"] += 1
            
        # 1. Confidence Profiling
        max_conf = float(np.max(cancer_prob))
        stats["average_max_conf"].append(max_conf)
        
        if max_conf < 0.1:
            stats["max_conf_under_10_percent"] += 1
        if max_conf < 0.01:
            stats["max_conf_under_1_percent"] += 1
            
        # 2. Thresholding & Blank Mask Check
        pred_10 = (cancer_prob >= 0.1).astype(np.int8)
        pred_50 = (cancer_prob >= 0.5).astype(np.int8)
        
        if np.sum(pred_10) == 0:
            stats["blank_predictions_at_10_percent"] += 1
            
        # 3. Raw Dice Calculation (only if patient actually has cancer)
        if has_cancer:
            stats["dice_at_10_percent"].append(dice_score(pred_10, gt_arr))
            stats["dice_at_50_percent"].append(dice_score(pred_50, gt_arr))

    print("\n📊 CONFIDENCE PROFILER")
    print(f"Average MAXIMUM confidence per scan: {np.mean(stats['average_max_conf'])*100:.2f}%")
    print(f"Scans where max confidence was < 10%:  {stats['max_conf_under_10_percent']} / {len(sample_files)}")
    print(f"Scans where max confidence was < 1%:   {stats['max_conf_under_1_percent']} / {len(sample_files)}")
    
    print("\n📊 MEMORY THRESHOLD IMPACT (Threshold = 10%)")
    print(f"Completely BLANK predictions:          {stats['blank_predictions_at_10_percent']} / {len(sample_files)}")
    
    print("\n📊 RAW DICE SCORES (On the {} patients with actual cancer)".format(stats['true_positives']))
    if stats['true_positives'] > 0:
        print(f"Mean Dice (threshold = 10%):           {np.mean(stats['dice_at_10_percent']):.4f}")
        print(f"Mean Dice (threshold = 50%):           {np.mean(stats['dice_at_50_percent']):.4f}")
    
    print("\n" + "="*60)
    if stats['max_conf_under_10_percent'] > (len(sample_files) * 0.5):
        print("🚨 DIAGNOSIS: MEMORY THRESHOLD BUG DETECTED!")
        print("The model is extremely underconfident due to domain shift.")
        print("Because max confidence is often < 10%, our 10% threshold clamped everything to 0.0!")
        print("We must lower the threshold in evaluate_pcnn.py to 0.01 or 0.001.")
    elif np.mean(stats['dice_at_10_percent']) < 0.05:
        print("🚨 DIAGNOSIS: TRUE DOMAIN SHIFT FAILURE DETECTED!")
        print("The model IS predicting confident tumors (>10%), but in completely the WRONG locations.")
        print("This is classic PI-CAI Domain Shift. The RUMC/ZGT weights are failing on the PCNN scanner.")
    else:
        print("✅ No catastrophic bugs found. Model is just performing moderately.")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--marksheet", type=str, required=True)
    args = parser.parse_args()
    
    diagnose_predictions(Path(args.val_dir), Path(args.gt_dir), Path(args.marksheet))
