import argparse
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def evaluate_metrics(val_dir: Path, gt_dir: Path, marksheet_path: Path):
    print("\n--- Running Official picai_eval Metrics (INTERNAL VALIDATION) ---")
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
    
    # FILTER FOR RUMC and ZGT instead of PCNN
    internal_patients = set(df[df['center'].isin(['RUMC', 'ZGT'])]['patient_id'].tolist())
        
    print(f"Scanning for predictions in: {val_dir}")
    all_y_det_files = sorted(list(val_dir.glob("*.nii.gz")))
    
    # Filter ONLY Internal cases
    internal_nii_files = []
    for f in all_y_det_files:
        patient_id = f.name.split("_")[0]
        if patient_id in internal_patients:
            internal_nii_files.append(f)
            
    print(f"Found {len(all_y_det_files)} total files, filtered down to {len(internal_nii_files)} RUMC/ZGT cases.")
    
    if len(internal_nii_files) == 0:
        print("No prediction files found! Did nnUNetv2_train --val run successfully?")
        return
        
    # Check if .npz continuous probabilities exist!
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
            
            # NOTE: No clipping is applied here!
            
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
        futures = {executor.submit(process_file, f): f for f in internal_nii_files}
        
        for i, future in enumerate(as_completed(futures), 1):
            y_det_files.append(future.result())
            if i % 50 == 0 or i == len(internal_nii_files):
                print(f"Extracted {i}/{len(internal_nii_files)} files...")
            
    y_det_files.sort(key=lambda x: str(x))
            
    if y_det_files[0].parent == prob_dir:
        print("Successfully extracted continuous probabilities! Your AUROC will be accurate.")
    else:
        print("⚠️ WARNING: .npz probability files not found! Using binary masks. Your AUROC will be artificially low.")
    
    if len(y_det_files) == 0:
        print("No prediction files found!")
        return
        
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
    print("🏆 FINAL INTERNAL (RUMC/ZGT) BASELINE SCORES 🏆")
    print("="*50)
    print(f"Patient-Level AUROC: {metrics.auroc:.4f}")
    print(f"Lesion-Level AP:     {metrics.AP:.4f}")
    if y_det_files[0].parent != prob_dir:
        print("\n⚠️ NOTE: These scores were calculated using BINARY MASKS because .npz files were missing.")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_dir", type=str, required=True, help="Path to nnUNet validation output folder")
    parser.add_argument("--gt_dir", type=str, required=True, help="Path to gt_segmentations folder in nnUNet_preprocessed")
    parser.add_argument("--marksheet", type=str, required=True, help="Path to marksheet.csv")
    args = parser.parse_args()
    
    evaluate_metrics(Path(args.val_dir), Path(args.gt_dir), Path(args.marksheet))
