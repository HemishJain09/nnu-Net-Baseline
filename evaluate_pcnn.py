import argparse
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def evaluate_metrics(val_dir: Path, gt_dir: Path, marksheet_path: Path):
    print("\n--- Running Official picai_eval Metrics ---")
    try:
        from picai_eval import evaluate
        from report_guided_annotation import extract_lesion_candidates
    except ImportError:
        print("Required libraries missing! Please run:")
        print("pip install picai_eval report_guided_annotation")
        return
        
    import pandas as pd
    df = pd.read_csv(marksheet_path)
    df['patient_id'] = df['patient_id'].astype(str)
    pcnn_patients = set(df[df['center'] == 'PCNN']['patient_id'].tolist())
        
    print(f"Scanning for predictions in: {val_dir}")
    all_y_det_files = sorted(list(val_dir.glob("*.nii.gz")))
    
    # Filter ONLY PCNN cases
    pcnn_nii_files = []
    for f in all_y_det_files:
        patient_id = f.name.split("_")[0]
        if patient_id in pcnn_patients:
            pcnn_nii_files.append(f)
            
    print(f"Found {len(all_y_det_files)} total files, filtered down to {len(pcnn_nii_files)} PCNN cases.")
    
    if len(pcnn_nii_files) == 0:
        print("No prediction files found! Did nnUNetv2_train --val run successfully?")
        return
        
    # Check if .npz continuous probabilities exist!
    # AUROC requires continuous probabilities. If we pass binary masks, AUROC is fundamentally broken.
    prob_dir = val_dir / "continuous_probabilities"
    prob_dir.mkdir(parents=True, exist_ok=True)
    
    y_det_files = []
    
    print("\n--- Extracting Continuous Probabilities for AUROC (Parallelized) ---")
    
    def process_file(nii_file):
        case_id = nii_file.name.replace(".nii.gz", "")
        npz_file = val_dir / f"{case_id}.npz"
        prob_nii = prob_dir / f"{case_id}.nii.gz"
        
        if prob_nii.exists():
            # Skip extraction if we already generated it previously!
            return prob_nii
            
        if npz_file.exists():
            # Load the softmax probabilities from nnU-Net
            data = np.load(npz_file)
            probs = data['probabilities'] # shape: (num_classes, Z, Y, X)
            
            # Make a strict copy and drop to float32 to guarantee it is mutable 
            # and to save even more memory!
            cancer_prob = np.array(probs[1], dtype=np.float32) # Class 1 (Cancer)
            
            # MEMORY LEAK FIX: Zero out ultra-low confidence background noise
            # This prevents picai_eval from generating millions of useless lesion candidates
            # and blowing up the RAM to 82 GB!
            cancer_prob[cancer_prob < 0.1] = 0.0
            
            # Read the geometry from the binary prediction .nii.gz
            ref_img = sitk.ReadImage(str(nii_file))
            
            # Convert probability array to NIfTI
            prob_img = sitk.GetImageFromArray(cancer_prob)
            prob_img.CopyInformation(ref_img)
            
            sitk.WriteImage(prob_img, str(prob_nii))
            return prob_nii
        else:
            # Fallback to binary mask (this will ruin AUROC, but it's safe)
            return nii_file
            
    # Run extraction in parallel using 4-8 threads depending on CPU
    import multiprocessing
    num_workers = min(8, multiprocessing.cpu_count() * 2)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_file, f): f for f in pcnn_nii_files}
        
        for i, future in enumerate(as_completed(futures), 1):
            y_det_files.append(future.result())
            if i % 50 == 0 or i == len(pcnn_nii_files):
                print(f"Extracted {i}/{len(pcnn_nii_files)} files...")
            
    # Sort files to ensure consistency
    y_det_files.sort(key=lambda x: str(x))
            
    if y_det_files[0].parent == prob_dir:
        print("Successfully extracted continuous probabilities! Your AUROC will be accurate.")
    else:
        print("⚠️ WARNING: .npz probability files not found! Using binary masks. Your AUROC will be artificially low (near 0.5) because binary masks do not contain confidence scores!")
    
    if len(y_det_files) == 0:
        print("No prediction .nii.gz files found! Did nnUNetv2_train --val run successfully?")
        return
        
    # Match the predictions exactly to the ground truth files in gt_segmentations
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
        
    print(f"Evaluating {len(valid_y_det)} predictions against ground truth...")
    
    metrics = evaluate(
        y_true=y_true_files,
        y_det=valid_y_det,
        subject_list=[Path(f).name.replace(".nii.gz", "") for f in y_true_files],
        num_parallel_calls=2,
        y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0]
    )
    
    print("\n" + "="*50)
    print("🏆 FINAL PCNN HOLDOUT SCORES 🏆")
    print("="*50)
    print(f"Patient-Level AUROC: {metrics.auroc:.4f}")
    print(f"Lesion-Level AP:     {metrics.AP:.4f}")
    if y_det_files[0].parent != prob_dir:
        print("\n⚠️ NOTE: These scores were calculated using BINARY MASKS because .npz files were missing.")
        print("To get your true AUROC, you MUST pass --save_probabilities to your predict command.")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_dir", type=str, required=True, help="Path to nnUNet validation output folder")
    parser.add_argument("--gt_dir", type=str, required=True, help="Path to gt_segmentations folder in nnUNet_preprocessed")
    parser.add_argument("--marksheet", type=str, required=True, help="Path to marksheet.csv")
    args = parser.parse_args()
    
    evaluate_metrics(Path(args.val_dir), Path(args.gt_dir), Path(args.marksheet))
