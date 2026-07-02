"""
===============================================================================
PI-CAI Pre-processed → nnUNet v2 Data Conversion Script
===============================================================================
Converts the PI-CAI pre-processed dataset into nnUNet v2's strict
Dataset500_PICAI format with 6 input channels:
  - Channel 0 (T2W):       from t2/
  - Channel 1 (ADC):       from adc_reg/
  - Channel 2 (HBV):       from hbv_reg/
  - Channel 3 (ZONAL_BG):  one-hot from zonal_masks/ (zone == 0)
  - Channel 4 (ZONAL_PZ):  one-hot from zonal_masks/ (zone == 1)
  - Channel 5 (ZONAL_TZ):  one-hot from zonal_masks/ (zone == 2)

Handles the "Missing Mask" trap: generates all-zeros NIfTI masks for the 207
benign patients who lack lesion annotations.

Usage:
  # Full dataset:
  python convert_picai_to_nnunet.py \
    --source_dir /path/to/PI-CAI_pre-processed \
    --nnunet_raw /path/to/nnUNet_raw

  # Sanity check (10 patients only):
  python convert_picai_to_nnunet.py \
    --source_dir /path/to/PI-CAI_pre-processed \
    --nnunet_raw /path/to/nnUNet_raw \
    --max_cases 10

Author: Auto-generated for PI-CAI nnUNet v2 pipeline
===============================================================================
"""

import os
import json
import argparse
import shutil
from pathlib import Path

import numpy as np
import nibabel as nib
from tqdm import tqdm


# ─── Configuration ───────────────────────────────────────────────────────────
DATASET_ID = 500
DATASET_NAME = "Dataset500_PICAI"


def one_hot_encode_zonal_mask(zonal_nifti_path: Path, output_dir: Path, case_id: str):
    """
    Load a zonal segmentation mask and split it into 3 binary one-hot channels.
    
    The PI-CAI zonal masks typically contain:
      0 = Background
      1 = Peripheral Zone (PZ)
      2 = Transition Zone (TZ)
    
    We create:
      {case_id}_0003.nii.gz  →  BG channel  (1 where label==0)
      {case_id}_0004.nii.gz  →  PZ channel  (1 where label==1)
      {case_id}_0005.nii.gz  →  TZ channel  (1 where label==2)
    """
    img = nib.load(str(zonal_nifti_path))
    data = img.get_fdata().astype(np.int8)
    affine = img.affine
    header = img.header
    
    # One-hot encode into 3 binary channels
    channels = {
        "_0003": (data == 0).astype(np.int8),   # Background
        "_0004": (data == 1).astype(np.int8),   # Peripheral Zone
        "_0005": (data == 2).astype(np.int8),   # Transition Zone
    }
    
    for suffix, channel_data in channels.items():
        out_path = output_dir / f"{case_id}{suffix}.nii.gz"
        out_img = nib.Nifti1Image(channel_data, affine, header)
        nib.save(out_img, str(out_path))


def generate_empty_mask(reference_nifti_path: Path, output_path: Path):
    """
    Generate an all-zeros NIfTI mask matching the geometry of a reference image.
    Used for the 207 benign/healthy patients who lack lesion annotations.
    """
    ref_img = nib.load(str(reference_nifti_path))
    empty_data = np.zeros(ref_img.shape, dtype=np.int8)
    empty_img = nib.Nifti1Image(empty_data, ref_img.affine, ref_img.header)
    nib.save(empty_img, str(output_path))


def convert_picai_to_nnunet(source_dir: Path, nnunet_raw: Path, max_cases: int = 0):
    """
    Main conversion function.
    
    Reads from PI-CAI pre-processed structure and creates:
      nnUNet_raw/Dataset500_PICAI/
        ├── dataset.json
        ├── imagesTr/
        │   ├── {caseID}_0000.nii.gz  (T2W)
        │   ├── {caseID}_0001.nii.gz  (ADC)
        │   ├── {caseID}_0002.nii.gz  (HBV)
        │   ├── {caseID}_0003.nii.gz  (Zonal BG)
        │   ├── {caseID}_0004.nii.gz  (Zonal PZ)
        │   ├── {caseID}_0005.nii.gz  (Zonal TZ)
        ├── labelsTr/
        │   ├── {caseID}.nii.gz       (Lesion mask)
    """
    # Define source subdirectories
    t2_dir = source_dir / "t2"
    adc_dir = source_dir / "adc_reg"
    hbv_dir = source_dir / "hbv_reg"
    zonal_dir = source_dir / "zonal_masks"
    lesion_dir = source_dir / "lesion_masks"
    
    # Define output directories
    dataset_dir = nnunet_raw / DATASET_NAME
    
    # Clean output directory if it exists to prevent leftover files from previous runs
    if dataset_dir.exists():
        print(f"🧹 Cleaning existing dataset directory: {dataset_dir}")
        shutil.rmtree(dataset_dir)
        
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    
    # Create directories
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Discover all cases from t2/ directory (authoritative source — always 1500)
    t2_files = sorted(list(t2_dir.glob("*.nii.gz")))
    print(f"Found {len(t2_files)} T2W scans in {t2_dir}")
    
    # Apply max_cases limit if specified (for sanity check mode)
    if max_cases > 0 and max_cases < len(t2_files):
        t2_files = t2_files[:max_cases]
        print(f"⚡ SANITY CHECK MODE: Processing only {max_cases} cases")
    
    # Track statistics
    stats = {
        "total_cases": 0,
        "masks_copied": 0,
        "masks_generated": 0,
        "missing_adc": [],
        "missing_hbv": [],
        "missing_zonal": [],
    }
    
    for t2_file in tqdm(t2_files, desc="Converting cases"):
        # Extract case ID: e.g., "10001_1000001" from "10001_1000001.nii.gz"
        case_id = t2_file.name.replace(".nii.gz", "")
        stats["total_cases"] += 1
        
        # ─── Channel 0: T2W ──────────────────────────────────────────────
        dest_t2 = images_dir / f"{case_id}_0000.nii.gz"
        if not dest_t2.exists():
            shutil.copy2(str(t2_file), str(dest_t2))
        
        # ─── Channel 1: ADC ──────────────────────────────────────────────
        src_adc = adc_dir / t2_file.name
        dest_adc = images_dir / f"{case_id}_0001.nii.gz"
        if src_adc.exists():
            if not dest_adc.exists():
                shutil.copy2(str(src_adc), str(dest_adc))
        else:
            stats["missing_adc"].append(case_id)
        
        # ─── Channel 2: HBV ──────────────────────────────────────────────
        src_hbv = hbv_dir / t2_file.name
        dest_hbv = images_dir / f"{case_id}_0002.nii.gz"
        if src_hbv.exists():
            if not dest_hbv.exists():
                shutil.copy2(str(src_hbv), str(dest_hbv))
        else:
            stats["missing_hbv"].append(case_id)
        
        # ─── Channels 3-5: Zonal Masks (One-Hot) ─────────────────────────
        src_zonal = zonal_dir / t2_file.name
        dest_zonal_bg = images_dir / f"{case_id}_0003.nii.gz"
        if src_zonal.exists():
            if not dest_zonal_bg.exists():  # Check if already done
                one_hot_encode_zonal_mask(src_zonal, images_dir, case_id)
        else:
            stats["missing_zonal"].append(case_id)
        
        # ─── Label: Lesion Mask ───────────────────────────────────────────
        src_mask = lesion_dir / t2_file.name
        dest_mask = labels_dir / f"{case_id}.nii.gz"
        
        if not dest_mask.exists():
            if src_mask.exists():
                # Lesion mask exists → copy directly (NO resampling!)
                shutil.copy2(str(src_mask), str(dest_mask))
                stats["masks_copied"] += 1
            else:
                # Missing mask → generate all-zeros (negative/benign case)
                generate_empty_mask(t2_file, dest_mask)
                stats["masks_generated"] += 1
    
    # ─── Generate dataset.json ────────────────────────────────────────────
    dataset_json = {
        "channel_names": {
            "0": "T2W",
            "1": "ADC",
            "2": "HBV",
            "3": "ZONAL_BG",
            "4": "ZONAL_PZ",
            "5": "ZONAL_TZ",
        },
        "labels": {
            "background": 0,
            "csPCa": 1,
        },
        "numTraining": stats["total_cases"],
        "file_ending": ".nii.gz",
    }
    
    json_path = dataset_dir / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=2)
    
    # ─── Print Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Total cases processed:    {stats['total_cases']}")
    print(f"Lesion masks copied:      {stats['masks_copied']}")
    print(f"Blank masks generated:    {stats['masks_generated']}")
    print(f"Missing ADC files:        {len(stats['missing_adc'])}")
    print(f"Missing HBV files:        {len(stats['missing_hbv'])}")
    print(f"Missing Zonal files:      {len(stats['missing_zonal'])}")
    print(f"\nOutput directory:          {dataset_dir}")
    print(f"dataset.json saved to:    {json_path}")
    print("=" * 60)
    
    if stats["missing_adc"] or stats["missing_hbv"]:
        print("\n⚠️  WARNING: Some modalities are missing!")
        if stats["missing_adc"]:
            print(f"  Missing ADC: {stats['missing_adc'][:5]}...")
        if stats["missing_hbv"]:
            print(f"  Missing HBV: {stats['missing_hbv'][:5]}...")
    
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert PI-CAI pre-processed data to nnUNet v2 format."
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Path to PI-CAI_pre-processed directory",
    )
    parser.add_argument(
        "--nnunet_raw",
        type=str,
        required=True,
        help="Path to nnUNet_raw directory (will be created if needed)",
    )
    parser.add_argument(
        "--max_cases",
        type=int,
        default=0,
        help="Max number of cases to process (0 = all). Use 10-15 for sanity check.",
    )
    
    args = parser.parse_args()
    
    source_dir = Path(args.source_dir)
    nnunet_raw = Path(args.nnunet_raw)
    
    # Validate source directory
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    if not (source_dir / "t2").exists():
        raise FileNotFoundError(f"Expected t2/ subdirectory in {source_dir}")
    
    convert_picai_to_nnunet(source_dir, nnunet_raw, max_cases=args.max_cases)
