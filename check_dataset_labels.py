import glob
import os
import blosc2
import numpy as np

def check_labels():
    print("="*50)
    print("🔍 DIAGNOSING DATASET LABELS (CANCER PREVALENCE)")
    print("="*50)
    
    cache_dir = '/content/nnUNet_preprocessed/Dataset500_PICAI/nnUNetPlans_3d_fullres'
    masks = sorted(glob.glob(f'{cache_dir}/*_seg.b2nd'))
    
    if len(masks) == 0:
        print("❌ No masks found! Check your path.")
        return
        
    print(f"Found {len(masks)} patient masks in cache. Scanning...")
    
    total_cancer_pixels = 0
    total_patients_with_cancer = 0
    
    for mask_path in masks:
        mask = blosc2.open(urlpath=mask_path, mode='r')[:]
        cancer_pixels = np.sum(mask == 1)
        
        if cancer_pixels > 0:
            total_patients_with_cancer += 1
            total_cancer_pixels += cancer_pixels
            
    print("\n📊 RESULTS:")
    print(f"Total Patients Scanned: {len(masks)}")
    print(f"Patients with ANY cancer pixels: {total_patients_with_cancer}")
    print(f"Total Cancer Pixels in entire dataset: {total_cancer_pixels}")
    
    if total_patients_with_cancer == 0:
        print("\n🚨 CRITICAL BUG DETECTED: THE 'EMPTY MASK' BUG!")
        print("There is absolutely zero cancer in your training dataset.")
        print("Your conversion script accidentally wiped out or ignored all lesion masks.")
        print("Because the network has never seen a cancer pixel in its life, it mathematically learned to output 0.0 everywhere.")
    else:
        print("\n✅ Dataset labels are healthy! Cancer exists in the dataset.")
        print(f"Cancer Prevalence: {total_patients_with_cancer / len(masks) * 100:.1f}% of patients have lesions.")

if __name__ == '__main__':
    check_labels()
