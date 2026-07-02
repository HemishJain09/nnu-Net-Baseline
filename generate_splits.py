"""
===============================================================================
Patient-Level GroupKFold Splits Generator for nnUNet v2
===============================================================================
Generates a custom splits_final.json that prevents data leakage from the 24
PI-CAI patients with longitudinal (multi-study) scans.

All studies belonging to the same patientID are grouped into the same fold,
ensuring no patient appears in both training and validation sets.

Output format (nnUNet v2 compatible):
  [
    {"train": ["10001_1000001", ...], "val": ["10008_1000008", ...]},
    {"train": [...], "val": [...]},
    ... (5 folds total)
  ]

Usage:
  python generate_splits.py \
    --nnunet_raw /path/to/nnUNet_raw \
    --nnunet_preprocessed /path/to/nnUNet_preprocessed

Author: Auto-generated for PI-CAI nnUNet v2 pipeline
===============================================================================
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.model_selection import GroupKFold


DATASET_NAME = "Dataset500_PICAI"
NUM_FOLDS = 5


def generate_splits(nnunet_raw: Path, nnunet_preprocessed: Path):
    """
    Generate patient-level GroupKFold splits and save as splits_final.json.
    
    Steps:
    1. Discover all case IDs from labelsTr/ directory
    2. Extract patientID from each case (first part of filename)
    3. Group cases by patientID
    4. Perform 5-fold GroupKFold split at patient level
    5. Save splits_final.json to the preprocessed directory
    """
    # Discover all case IDs from labelsTr
    labels_dir = nnunet_raw / DATASET_NAME / "labelsTr"
    
    if not labels_dir.exists():
        raise FileNotFoundError(
            f"labelsTr directory not found at {labels_dir}. "
            "Run convert_picai_to_nnunet.py first."
        )
    
    label_files = sorted(list(labels_dir.glob("*.nii.gz")))
    case_ids = [f.name.replace(".nii.gz", "") for f in label_files]
    
    print(f"Found {len(case_ids)} cases in {labels_dir}")
    
    # Extract patient IDs and group studies
    patient_groups = defaultdict(list)
    case_to_patient = {}
    
    for case_id in case_ids:
        # Filename convention: patientID_studyID (e.g., 10001_1000001)
        patient_id = case_id.split("_")[0]
        patient_groups[patient_id].append(case_id)
        case_to_patient[case_id] = patient_id
    
    # Identify longitudinal patients (multiple studies)
    longitudinal = {pid: studies for pid, studies in patient_groups.items() 
                    if len(studies) > 1}
    
    print(f"Unique patients: {len(patient_groups)}")
    print(f"Longitudinal patients (>1 study): {len(longitudinal)}")
    for pid, studies in sorted(longitudinal.items()):
        print(f"  Patient {pid}: {studies}")
    
    # Prepare arrays for GroupKFold
    # Each case gets assigned its patient's group ID
    unique_patients = sorted(patient_groups.keys())
    patient_to_group_id = {pid: idx for idx, pid in enumerate(unique_patients)}
    
    groups = np.array([patient_to_group_id[case_to_patient[c]] for c in case_ids])
    X = np.arange(len(case_ids))  # Dummy features
    y = np.zeros(len(case_ids))   # Dummy labels (not used by GroupKFold)
    
    # Perform GroupKFold
    gkf = GroupKFold(n_splits=NUM_FOLDS)
    
    splits = []
    for fold_idx, (train_indices, val_indices) in enumerate(gkf.split(X, y, groups)):
        train_cases = [case_ids[i] for i in train_indices]
        val_cases = [case_ids[i] for i in val_indices]
        
        splits.append({
            "train": sorted(train_cases),
            "val": sorted(val_cases),
        })
        
        # Verify no patient leakage
        train_patients = set(c.split("_")[0] for c in train_cases)
        val_patients = set(c.split("_")[0] for c in val_cases)
        overlap = train_patients & val_patients
        
        print(f"\nFold {fold_idx}:")
        print(f"  Train: {len(train_cases)} cases ({len(train_patients)} patients)")
        print(f"  Val:   {len(val_cases)} cases ({len(val_patients)} patients)")
        
        if overlap:
            raise RuntimeError(
                f"DATA LEAKAGE DETECTED in Fold {fold_idx}! "
                f"Overlapping patients: {overlap}"
            )
        else:
            print(f"  ✅ No patient leakage detected")
    
    # Save splits_final.json to the preprocessed dataset directory
    output_dir = nnunet_preprocessed / DATASET_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "splits_final.json"
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"SPLITS GENERATED SUCCESSFULLY")
    print(f"{'=' * 60}")
    print(f"Output: {output_path}")
    print(f"Folds:  {NUM_FOLDS}")
    print(f"Total cases: {len(case_ids)}")
    print(f"{'=' * 60}")
    
    return splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate patient-level GroupKFold splits for nnUNet v2."
    )
    parser.add_argument(
        "--nnunet_raw",
        type=str,
        required=True,
        help="Path to nnUNet_raw directory",
    )
    parser.add_argument(
        "--nnunet_preprocessed",
        type=str,
        required=True,
        help="Path to nnUNet_preprocessed directory",
    )
    
    args = parser.parse_args()
    generate_splits(Path(args.nnunet_raw), Path(args.nnunet_preprocessed))
