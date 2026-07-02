"""
===============================================================================
Dummy Data Generator for Local Testing
===============================================================================
Creates realistic dummy NIfTI files for 5 patients (including 1 longitudinal
patient and 1 patient without a lesion mask) to test the full pipeline locally.

Usage:
  python create_test_data.py --output_dir ./test_data
===============================================================================
"""

import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
import pandas as pd


# 5 unique patients, patient 10001 has 2 studies (longitudinal)
# Patient 10005 will NOT have a lesion mask (negative case)
TEST_CASES = [
    "10001_1000001",  # Patient 10001, study 1
    "10001_1000002",  # Patient 10001, study 2 (LONGITUDINAL - same patient!)
    "10002_1000003",
    "10003_1000004",
    "10004_1000005",
    "10005_1000006",  # NO lesion mask (negative case)
]

NEGATIVE_CASES = {"10005_1000006"}  # Patients without lesion masks

# Realistic PI-CAI image dimensions
IMAGE_SHAPE = (20, 320, 320)
SPACING = (3.0, 0.5, 0.5)
AFFINE = np.diag([SPACING[2], SPACING[1], SPACING[0], 1.0])


def create_nifti(data: np.ndarray, output_path: Path):
    """Save a numpy array as a compressed NIfTI file."""
    img = nib.Nifti1Image(data, AFFINE)
    img.header.set_zooms(SPACING)
    nib.save(img, str(output_path))


def generate_test_data(output_dir: Path):
    """Generate realistic test data matching PI-CAI pre-processed structure."""
    
    # Create directory structure
    dirs = {
        "t2": output_dir / "t2",
        "adc_reg": output_dir / "adc_reg",
        "hbv_reg": output_dir / "hbv_reg",
        "zonal_masks": output_dir / "zonal_masks",
        "lesion_masks": output_dir / "lesion_masks",
        "clinical_information": output_dir / "clinical_information",
    }
    
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating test data for {len(TEST_CASES)} cases...")
    
    for case_id in TEST_CASES:
        filename = f"{case_id}.nii.gz"
        
        # T2W: High signal range (0-2000)
        t2_data = np.random.randint(0, 2000, IMAGE_SHAPE, dtype=np.int16)
        create_nifti(t2_data, dirs["t2"] / filename)
        
        # ADC: Typical ADC range (0-3000 x 10^-6 mm²/s)
        adc_data = np.random.randint(0, 3000, IMAGE_SHAPE, dtype=np.int16)
        create_nifti(adc_data, dirs["adc_reg"] / filename)
        
        # HBV: DWI high b-value (0-1500)
        hbv_data = np.random.randint(0, 1500, IMAGE_SHAPE, dtype=np.int16)
        create_nifti(hbv_data, dirs["hbv_reg"] / filename)
        
        # Zonal mask: 0=BG, 1=PZ, 2=TZ
        zonal_data = np.zeros(IMAGE_SHAPE, dtype=np.int8)
        # Create a simple prostate-like region
        z_mid, y_mid, x_mid = [s // 2 for s in IMAGE_SHAPE]
        # TZ (inner zone)
        zonal_data[z_mid-3:z_mid+3, y_mid-30:y_mid+30, x_mid-25:x_mid+25] = 2
        # PZ (outer zone)
        zonal_data[z_mid-4:z_mid+4, y_mid-50:y_mid+50, x_mid-40:x_mid+40] = np.where(
            zonal_data[z_mid-4:z_mid+4, y_mid-50:y_mid+50, x_mid-40:x_mid+40] == 0,
            1, zonal_data[z_mid-4:z_mid+4, y_mid-50:y_mid+50, x_mid-40:x_mid+40]
        )
        create_nifti(zonal_data, dirs["zonal_masks"] / filename)
        
        # Lesion mask: Binary (skip for negative cases)
        if case_id not in NEGATIVE_CASES:
            lesion_data = np.zeros(IMAGE_SHAPE, dtype=np.int8)
            # Small lesion blob
            lesion_data[z_mid-1:z_mid+1, y_mid-10:y_mid+10, x_mid-10:x_mid+10] = 1
            create_nifti(lesion_data, dirs["lesion_masks"] / filename)
        else:
            print(f"  ⚠️  Skipping lesion mask for {case_id} (negative case)")
    
    # Create a minimal marksheet.csv
    rows = []
    for case_id in TEST_CASES:
        patient_id, study_id = case_id.split("_")
        rows.append({
            "patient_id": patient_id,
            "study_id": study_id,
            "scanner_manufacturer": "Siemens",
            "scanner_model_name": "Skyra",
            "psa": round(np.random.uniform(1.0, 20.0), 2),
            "psad": round(np.random.uniform(0.05, 0.5), 3),
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(dirs["clinical_information"] / "marksheet.csv", index=False)
    
    print(f"\n✅ Generated test data at: {output_dir}")
    print(f"   - {len(TEST_CASES)} cases total")
    print(f"   - {len(NEGATIVE_CASES)} negative cases (no lesion mask)")
    print(f"   - 1 longitudinal patient (10001 with 2 studies)")
    print(f"   - Modalities: t2, adc_reg, hbv_reg, zonal_masks, lesion_masks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dummy PI-CAI test data.")
    parser.add_argument(
        "--output_dir", type=str, default="./test_data",
        help="Output directory for test data"
    )
    args = parser.parse_args()
    generate_test_data(Path(args.output_dir))
