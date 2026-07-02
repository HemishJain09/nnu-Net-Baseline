"""
===============================================================================
PI-CAI Pre-processed → nnUNet v2 Data Conversion Script
===============================================================================
Converts the PI-CAI pre-processed dataset into nnUNet v2's strict
Dataset500_PICAI format with 6 input channels.

Strict Geometry Alignment:
nnU-Net strictly requires all channels for a given patient to have identical
shape, spacing, and direction. Some PI-CAI patients have mismatched zonal
masks (e.g. 384x384 mask vs 640x640 MRI).
This script loads the T2W scan as a strict reference, and uses SimpleITK
to aggressively resample any mismatched channel to align perfectly with the T2W.

Channels:
  - Channel 0 (T2W):       from t2/
  - Channel 1 (ADC):       from adc_reg/
  - Channel 2 (HBV):       from hbv_reg/
  - Channel 3 (ZONAL_BG):  one-hot from zonal_masks/ (zone == 0)
  - Channel 4 (ZONAL_PZ):  one-hot from zonal_masks/ (zone == 1)
  - Channel 5 (ZONAL_TZ):  one-hot from zonal_masks/ (zone == 2)
"""

import os
import json
import argparse
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

DATASET_NAME = "Dataset500_PICAI"

def align_image_to_reference(img: sitk.Image, ref_img: sitk.Image, is_mask: bool) -> sitk.Image:
    """
    Checks if img perfectly matches ref_img geometry. If not, resamples it.
    Uses NearestNeighbor for masks (categorical), and BSpline for MRIs (continuous).
    """
    if (img.GetSize() != ref_img.GetSize() or 
        not np.allclose(img.GetSpacing(), ref_img.GetSpacing(), atol=1e-5) or
        not np.allclose(img.GetDirection(), ref_img.GetDirection(), atol=1e-5)):
        
        interpolator = sitk.sitkNearestNeighbor if is_mask else sitk.sitkBSpline
        resampled_img = sitk.Resample(img, ref_img, sitk.Transform(), interpolator, 0.0, img.GetPixelID())
        return resampled_img
    
    # Even if properties match closely, forcefully sync metadata to prevent nnU-Net float precision warnings
    img.CopyInformation(ref_img)
    return img

def one_hot_encode_zonal_mask(zonal_img: sitk.Image, ref_img: sitk.Image, output_dir: Path, case_id: str):
    aligned_zonal = align_image_to_reference(zonal_img, ref_img, is_mask=True)
    zonal_arr = sitk.GetArrayFromImage(aligned_zonal)
    
    channels = {
        "_0003": (zonal_arr == 0).astype(np.int8),
        "_0004": (zonal_arr == 1).astype(np.int8),
        "_0005": (zonal_arr == 2).astype(np.int8),
    }
    
    for suffix, channel_data in channels.items():
        out_path = output_dir / f"{case_id}{suffix}.nii.gz"
        out_img = sitk.GetImageFromArray(channel_data)
        out_img.CopyInformation(ref_img)
        sitk.WriteImage(out_img, str(out_path))

def generate_empty_mask(ref_img: sitk.Image, output_path: Path):
    empty_arr = np.zeros(ref_img.GetSize()[::-1], dtype=np.int8)
    empty_img = sitk.GetImageFromArray(empty_arr)
    empty_img.CopyInformation(ref_img)
    sitk.WriteImage(empty_img, str(output_path))

def process_lesion_mask(src_path: Path, ref_img: sitk.Image, dest_path: Path):
    img = sitk.ReadImage(str(src_path))
    aligned_img = align_image_to_reference(img, ref_img, is_mask=True)
    
    arr = sitk.GetArrayFromImage(aligned_img)
    if np.any(arr > 1) or np.any(arr < 0):
        arr = (arr > 0).astype(np.int8)
        bin_img = sitk.GetImageFromArray(arr)
        bin_img.CopyInformation(ref_img)
        sitk.WriteImage(bin_img, str(dest_path))
    else:
        # Array is already binary [0, 1], just save the aligned image
        arr = arr.astype(np.int8)
        bin_img = sitk.GetImageFromArray(arr)
        bin_img.CopyInformation(ref_img)
        sitk.WriteImage(bin_img, str(dest_path))

def process_mri_channel(src_path: Path, ref_img: sitk.Image, dest_path: Path):
    img = sitk.ReadImage(str(src_path))
    aligned_img = align_image_to_reference(img, ref_img, is_mask=False)
    sitk.WriteImage(aligned_img, str(dest_path))


def convert_picai_to_nnunet(source_dir: Path, nnunet_raw: Path, max_cases: int = 0):
    t2_dir = source_dir / "t2"
    adc_dir = source_dir / "adc_reg"
    hbv_dir = source_dir / "hbv_reg"
    zonal_dir = source_dir / "zonal_masks"
    lesion_dir = source_dir / "lesion_masks"
    
    dataset_dir = nnunet_raw / DATASET_NAME
    if dataset_dir.exists():
        print(f"🧹 Cleaning existing dataset directory: {dataset_dir}")
        shutil.rmtree(dataset_dir)
        
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    t2_files = sorted(list(t2_dir.glob("*.nii.gz")))
    print(f"Found {len(t2_files)} T2W scans in {t2_dir}")
    
    if max_cases > 0 and max_cases < len(t2_files):
        t2_files = t2_files[:max_cases]
        print(f"⚡ SANITY CHECK MODE: Processing only {max_cases} cases")
    
    stats = {
        "total_cases": 0, "masks_copied": 0, "masks_generated": 0,
        "missing_adc": [], "missing_hbv": [], "missing_zonal": [],
    }
    
    for t2_file in tqdm(t2_files, desc="Converting cases"):
        case_id = t2_file.name.replace(".nii.gz", "")
        stats["total_cases"] += 1
        
        # 0. T2W (Reference Image)
        ref_img = sitk.ReadImage(str(t2_file))
        dest_t2 = images_dir / f"{case_id}_0000.nii.gz"
        sitk.WriteImage(ref_img, str(dest_t2))
        
        # 1. ADC
        src_adc = adc_dir / t2_file.name
        dest_adc = images_dir / f"{case_id}_0001.nii.gz"
        if src_adc.exists():
            process_mri_channel(src_adc, ref_img, dest_adc)
        else:
            stats["missing_adc"].append(case_id)
        
        # 2. HBV
        src_hbv = hbv_dir / t2_file.name
        dest_hbv = images_dir / f"{case_id}_0002.nii.gz"
        if src_hbv.exists():
            process_mri_channel(src_hbv, ref_img, dest_hbv)
        else:
            stats["missing_hbv"].append(case_id)
        
        # 3-5. Zonal Masks
        src_zonal = zonal_dir / t2_file.name
        dest_zonal_bg = images_dir / f"{case_id}_0003.nii.gz"
        if src_zonal.exists():
            if not dest_zonal_bg.exists():
                zonal_img = sitk.ReadImage(str(src_zonal))
                one_hot_encode_zonal_mask(zonal_img, ref_img, images_dir, case_id)
        else:
            stats["missing_zonal"].append(case_id)
        
        # Label: Lesion Mask
        src_mask = lesion_dir / t2_file.name
        dest_mask = labels_dir / f"{case_id}.nii.gz"
        if src_mask.exists():
            process_lesion_mask(src_mask, ref_img, dest_mask)
            stats["masks_copied"] += 1
        else:
            generate_empty_mask(ref_img, dest_mask)
            stats["masks_generated"] += 1
            
    dataset_json = {
        "channel_names": {
            "0": "T2W", "1": "ADC", "2": "HBV",
            "3": "ZONAL_BG", "4": "ZONAL_PZ", "5": "ZONAL_TZ"
        },
        "labels": {"background": 0, "csPCa": 1},
        "numTraining": stats["total_cases"],
        "file_ending": ".nii.gz",
    }
    
    json_path = dataset_dir / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=2)
    
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print(f"Total cases processed:    {stats['total_cases']}")
    print(f"Lesion masks copied:      {stats['masks_copied']}")
    print(f"Blank masks generated:    {stats['masks_generated']}")
    print(f"Missing ADC files:        {len(stats['missing_adc'])}")
    print(f"Missing HBV files:        {len(stats['missing_hbv'])}")
    print(f"Missing Zonal files:      {len(stats['missing_zonal'])}")
    print("=" * 60)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, required=True)
    parser.add_argument("--nnunet_raw", type=str, required=True)
    parser.add_argument("--max_cases", type=int, default=0)
    args = parser.parse_args()
    convert_picai_to_nnunet(Path(args.source_dir), Path(args.nnunet_raw), max_cases=args.max_cases)
