"""
===============================================================================
Patient-Level Splits Generator for nnUNet v2
===============================================================================
Generates a custom splits_final.json.

Mode: Domain Adaptation (Pure Holdout)
We train/validate ONLY on the specified `--train_centers`.
Any center not in `--train_centers` is COMPLETELY excluded from the splits.
This leaves the third hospital as a pure, unseen holdout test set.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

DATASET_NAME = "Dataset500_PICAI"
NUM_FOLDS = 5

def generate_splits(nnunet_raw: Path, nnunet_preprocessed: Path, marksheet_path: Path, train_centers: list, bootcamp: bool = False):
    case_ids = []
    
    preprocessed_dir = nnunet_preprocessed / DATASET_NAME / "nnUNetPlans_3d_fullres"
    if preprocessed_dir.exists():
        preprocessed_files = list(preprocessed_dir.glob("*.pkl"))
        if preprocessed_files:
            preprocessed_files = sorted(preprocessed_files)
            case_ids = [f.stem for f in preprocessed_files if not f.stem.endswith('_seg')]
            print(f"Loaded {len(case_ids)} case IDs from preprocessed directory.")

    if not case_ids:
        labels_dir = nnunet_raw / DATASET_NAME / "labelsTr"
        if not labels_dir.exists():
            raise FileNotFoundError(f"Neither preprocessed data nor labelsTr directory found.")
        
        label_files = sorted(list(labels_dir.glob("*.nii.gz")))
        case_ids = [f.name.replace(".nii.gz", "") for f in label_files]
        print(f"Loaded {len(case_ids)} case IDs from raw labelsTr directory.")
        
    if not case_ids:
        raise ValueError("No case IDs found in either preprocessed or raw directories!")
        
    print(f"Total cases in dataset folder: {len(case_ids)}")
    
    df = pd.read_csv(marksheet_path)
    df['patient_id'] = df['patient_id'].astype(str)
    patient_to_center = dict(zip(df['patient_id'], df['center']))
    patient_to_cancer = dict(zip(df['patient_id'], df['case_csPCa']))
    
    # Filter cases by train_centers and bootcamp mode
    valid_cases = []
    case_to_patient = {}
    patient_groups = defaultdict(list)
    
    for case in case_ids:
        pid = case.split("_")[0]
        center = patient_to_center.get(pid, "UNKNOWN")
        has_cancer = patient_to_cancer.get(pid, "NO") == "YES"
        
        if center in train_centers:
            # If Bootcamp is ON, completely drop healthy patients
            if bootcamp and not has_cancer:
                continue
                
            valid_cases.append(case)
            case_to_patient[case] = pid
            patient_groups[pid].append(case)
            
    print(f"\n--- Filtering for Training Centers: {train_centers} ---")
    if bootcamp:
        print(f"🚨 BOOTCAMP MODE ACTIVE: Excluding all healthy patients! 🚨")
    print(f"Cases kept for Training/Validation: {len(valid_cases)}")
    print(f"Cases completely excluded: {len(case_ids) - len(valid_cases)}")
    
    unique_patients = sorted(patient_groups.keys())
    patient_to_group_id = {pid: idx for idx, pid in enumerate(unique_patients)}
    groups = np.array([patient_to_group_id[case_to_patient[c]] for c in valid_cases])
    X = np.arange(len(valid_cases))
    y = np.zeros(len(valid_cases))
    
    splits = []
    gkf = GroupKFold(n_splits=NUM_FOLDS)
    for fold_idx, (train_indices, val_indices) in enumerate(gkf.split(X, y, groups)):
        train_cases = [valid_cases[i] for i in train_indices]
        val_cases = [valid_cases[i] for i in val_indices]
        splits.append({
            "train": sorted(train_cases),
            "val": sorted(val_cases),
        })
        print(f"\nFold {fold_idx}:")
        print(f"  Train: {len(train_cases)} cases")
        print(f"  Val:   {len(val_cases)} cases")
        
        # Verify no patient leakage
        train_patients = set(c.split("_")[0] for c in train_cases)
        val_patients = set(c.split("_")[0] for c in val_cases)
        if train_patients & val_patients:
            raise RuntimeError(f"DATA LEAKAGE DETECTED!")
            
    # Save splits_final.json
    output_dir = nnunet_preprocessed / DATASET_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "splits_final.json"
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"SPLITS GENERATED SUCCESSFULLY")
    print(f"Output: {output_path}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nnunet_raw", type=str, required=True)
    parser.add_argument("--nnunet_preprocessed", type=str, required=True)
    parser.add_argument("--marksheet", type=str, required=True, help="Path to marksheet.csv")
    parser.add_argument("--train_centers", type=str, nargs='+', required=True, help="Centers to include in training (e.g. RUMC ZGT)")
    parser.add_argument("--bootcamp", action="store_true", help="Bootcamp mode: Only include patients with cancer")
    args = parser.parse_args()
    
    generate_splits(Path(args.nnunet_raw), Path(args.nnunet_preprocessed), Path(args.marksheet), args.train_centers, args.bootcamp)
