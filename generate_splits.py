"""
===============================================================================
Patient-Level Splits Generator for nnUNet v2
===============================================================================
Generates a custom splits_final.json that prevents data leakage.

Modes:
1. Default: Standard 5-fold GroupKFold.
2. Domain Adaptation (LOCO): If --marksheet is provided, generates N folds 
   where N is the number of centers. Each fold trains on N-1 centers and 
   validates on the 1 holdout center.

Output format (nnUNet v2 compatible):
  [
    {"train": ["10001_1000001", ...], "val": ["10008_1000008", ...]},
    {"train": [...], "val": [...]},
    ...
  ]
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

def generate_splits(nnunet_raw: Path, nnunet_preprocessed: Path, marksheet_path: Path = None):
    labels_dir = nnunet_raw / DATASET_NAME / "labelsTr"
    if not labels_dir.exists():
        raise FileNotFoundError(f"labelsTr directory not found at {labels_dir}.")
    
    label_files = sorted(list(labels_dir.glob("*.nii.gz")))
    case_ids = [f.name.replace(".nii.gz", "") for f in label_files]
    print(f"Found {len(case_ids)} cases in {labels_dir}")
    
    patient_groups = defaultdict(list)
    case_to_patient = {}
    for case_id in case_ids:
        patient_id = case_id.split("_")[0]
        patient_groups[patient_id].append(case_id)
        case_to_patient[case_id] = patient_id
    
    splits = []
    
    if marksheet_path:
        print(f"\n--- Running Domain Adaptation (LOCO) Splits ---")
        print(f"Reading metadata from {marksheet_path}")
        df = pd.read_csv(marksheet_path)
        
        # Map patient_id to center
        # Assuming patient_id in CSV is integer or string matching our extracted patient_id
        df['patient_id'] = df['patient_id'].astype(str)
        patient_to_center = dict(zip(df['patient_id'], df['center']))
        
        # Find unique centers in our actual dataset cases
        present_centers = set()
        case_to_center = {}
        for case in case_ids:
            pid = case_to_patient[case]
            center = patient_to_center.get(pid, "UNKNOWN")
            case_to_center[case] = center
            present_centers.add(center)
            
        if "UNKNOWN" in present_centers:
            raise ValueError("Some patients in the dataset were not found in the marksheet.csv!")
            
        unique_centers = sorted(list(present_centers))
        print(f"Found {len(unique_centers)} clinical centers: {unique_centers}")
        
        for fold_idx, holdout_center in enumerate(unique_centers):
            train_cases = []
            val_cases = []
            
            for case in case_ids:
                if case_to_center[case] == holdout_center:
                    val_cases.append(case)
                else:
                    train_cases.append(case)
                    
            splits.append({
                "train": sorted(train_cases),
                "val": sorted(val_cases),
            })
            
            print(f"\nFold {fold_idx}: Holdout Center [{holdout_center}]")
            print(f"  Train: {len(train_cases)} cases")
            print(f"  Val:   {len(val_cases)} cases")
    else:
        print(f"\n--- Running Standard {NUM_FOLDS}-Fold GroupKFold Splits ---")
        unique_patients = sorted(patient_groups.keys())
        patient_to_group_id = {pid: idx for idx, pid in enumerate(unique_patients)}
        groups = np.array([patient_to_group_id[case_to_patient[c]] for c in case_ids])
        X = np.arange(len(case_ids))
        y = np.zeros(len(case_ids))
        
        gkf = GroupKFold(n_splits=NUM_FOLDS)
        for fold_idx, (train_indices, val_indices) in enumerate(gkf.split(X, y, groups)):
            train_cases = [case_ids[i] for i in train_indices]
            val_cases = [case_ids[i] for i in val_indices]
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
            overlap = train_patients & val_patients
            if overlap:
                raise RuntimeError(f"DATA LEAKAGE DETECTED! Overlapping patients: {overlap}")
    
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
    parser.add_argument("--marksheet", type=str, default=None, help="Path to marksheet.csv for Domain Adaptation splits")
    args = parser.parse_args()
    
    marksheet_path = Path(args.marksheet) if args.marksheet else None
    generate_splits(Path(args.nnunet_raw), Path(args.nnunet_preprocessed), marksheet_path)
