import argparse
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import GroupKFold

def get_pure_internal_patients(gt_dir: Path, marksheet_path: Path, splits_json: Path):
    print("="*60)
    print("🧬 RECREATING PHASE 1 SPLITS TO FIND DATA LEAKAGE...")
    
    df = pd.read_csv(marksheet_path)
    df['patient_id'] = df['patient_id'].astype(str)
    patient_to_center = dict(zip(df['patient_id'], df['center']))
    patient_to_cancer = dict(zip(df['patient_id'], df['case_csPCa']))
    
    # 1. Get all cases exactly as generate_splits.py did
    label_files = sorted(list(gt_dir.glob("*.nii.gz")))
    case_ids = [f.name.replace(".nii.gz", "") for f in label_files]
    
    # 2. Recreate Phase 1 (Bootcamp) Valid Cases
    valid_cases = []
    case_to_patient = {}
    patient_groups = defaultdict(list)
    
    for case in case_ids:
        pid = case.split("_")[0]
        center = patient_to_center.get(pid, "UNKNOWN")
        has_cancer = patient_to_cancer.get(pid, "NO") == "YES"
        
        if center in ["RUMC", "ZGT"] and has_cancer:
            valid_cases.append(case)
            case_to_patient[case] = pid
            patient_groups[pid].append(case)
            
    # 3. Recreate Phase 1 Folds
    unique_patients = sorted(patient_groups.keys())
    patient_to_group_id = {pid: idx for idx, pid in enumerate(unique_patients)}
    groups = np.array([patient_to_group_id[case_to_patient[c]] for c in valid_cases])
    X = np.arange(len(valid_cases))
    y = np.zeros(len(valid_cases))
    
    gkf = GroupKFold(n_splits=5)
    phase1_train_patients = set()
    
    # We only care about Fold 0 since that's what we trained on
    for fold_idx, (train_indices, val_indices) in enumerate(gkf.split(X, y, groups)):
        if fold_idx == 0:
            train_cases = [valid_cases[i] for i in train_indices]
            phase1_train_patients = set(c.split("_")[0] for c in train_cases)
            break
            
    # 4. Load Phase 2 Validation Patients from splits_final.json
    with open(splits_json, 'r') as f:
        splits = json.load(f)
    phase2_val_cases = splits[0]["val"]
    phase2_val_patients = set(c.split("_")[0] for c in phase2_val_cases)
    
    # 5. Calculate Pure Unseen Patients
    pure_val_patients = phase2_val_patients - phase1_train_patients
    leaked_patients = phase2_val_patients.intersection(phase1_train_patients)
    
    print(f"Phase 2 Val Patients: {len(phase2_val_patients)}")
    print(f"Leaked Patients (Seen in Phase 1): {len(leaked_patients)}")
    print(f"Pure Unseen Patients: {len(pure_val_patients)}")
    print("="*60)
    
    return pure_val_patients

def evaluate_metrics(val_dir: Path, gt_dir: Path, marksheet_path: Path, splits_json: Path):
    try:
        from picai_eval import evaluate
        from report_guided_annotation import extract_lesion_candidates
    except ImportError:
        print("Required libraries missing! Please run:")
        print("pip install picai_eval report_guided_annotation")
        return
        
    pure_patients = get_pure_internal_patients(gt_dir, marksheet_path, splits_json)
        
    print(f"\nScanning for predictions in: {val_dir}")
    all_y_det_files = sorted(list(val_dir.glob("*.nii.gz")))
    
    pure_nii_files = []
    for f in all_y_det_files:
        patient_id = f.name.split("_")[0]
        if patient_id in pure_patients:
            pure_nii_files.append(f)
            
    print(f"Filtered down to {len(pure_nii_files)} Pure Unseen RUMC/ZGT cases.")
    
    if len(pure_nii_files) == 0:
        print("No prediction files found! Did nnUNetv2_train --val run successfully?")
        return
        
    prob_dir = val_dir / "continuous_probabilities"
    prob_dir.mkdir(parents=True, exist_ok=True)
    
    y_det_files = []
    
    print("\n--- Extracting Continuous Probabilities for AUROC (Parallelized) ---")
    
    def process_file(nii_file):
        case_id = nii_file.name.replace(".nii.gz", "")
        npz_file = val_dir / f"{case_id}.npz"
        prob_nii = prob_dir / f"{case_id}.nii.gz"
        
        if prob_nii.exists():
            return prob_nii
            
        if npz_file.exists():
            data = np.load(npz_file)
            probs = data['probabilities']
            cancer_prob = np.array(probs[1], dtype=np.float32)
            
            ref_img = sitk.ReadImage(str(nii_file))
            prob_img = sitk.GetImageFromArray(cancer_prob)
            prob_img.CopyInformation(ref_img)
            
            sitk.WriteImage(prob_img, str(prob_nii))
            return prob_nii
        else:
            return nii_file
            
    import multiprocessing
    num_workers = min(8, multiprocessing.cpu_count() * 2)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_file, f): f for f in pure_nii_files}
        
        for i, future in enumerate(as_completed(futures), 1):
            y_det_files.append(future.result())
            if i % 50 == 0 or i == len(pure_nii_files):
                print(f"Extracted {i}/{len(pure_nii_files)} files...")
            
    y_det_files.sort(key=lambda x: str(x))
            
    y_true_files = []
    valid_y_det = []
    
    for det_file in y_det_files:
        gt_file = gt_dir / det_file.name
        if gt_file.exists():
            y_true_files.append(str(gt_file))
            valid_y_det.append(str(det_file))
        else:
            print(f"Warning: Ground truth for {det_file.name} not found in gt_segmentations. Skipping.")
            
    if len(y_true_files) == 0:
        print(f"No ground truth files found in {gt_dir}")
        return
        
    print(f"Evaluating {len(valid_y_det)} pure predictions against ground truth...")
    
    metrics = evaluate(
        y_true=y_true_files,
        y_det=valid_y_det,
        subject_list=[Path(f).name.replace(".nii.gz", "") for f in y_true_files],
        num_parallel_calls=2,
        y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0]
    )
    
    print("\n" + "="*50)
    print("🏆 MATHEMATICALLY PURE INTERNAL BASELINE SCORES 🏆")
    print("="*50)
    print(f"Patient-Level AUROC: {metrics.auroc:.4f}")
    print(f"Lesion-Level AP:     {metrics.AP:.4f}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_dir", type=str, required=True, help="Path to nnUNet validation output folder")
    parser.add_argument("--gt_dir", type=str, required=True, help="Path to gt_segmentations folder in nnUNet_preprocessed")
    parser.add_argument("--marksheet", type=str, required=True, help="Path to marksheet.csv")
    parser.add_argument("--splits", type=str, required=True, help="Path to splits_final.json (Phase 2)")
    args = parser.parse_args()
    
    evaluate_metrics(Path(args.val_dir), Path(args.gt_dir), Path(args.marksheet), Path(args.splits))
