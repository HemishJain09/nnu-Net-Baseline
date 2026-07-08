import argparse
from pathlib import Path

def evaluate_metrics(val_dir: Path, gt_dir: Path):
    print("\n--- Running Official picai_eval Metrics ---")
    try:
        from picai_eval import evaluate
    except ImportError:
        print("picai_eval not installed. Please run: pip install picai_eval")
        return
        
    print(f"Scanning for predictions in: {val_dir}")
    y_det_files = sorted(list(val_dir.glob("*.nii.gz")))
    
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
        subject_list=[Path(f).name.replace(".nii.gz", "") for f in y_true_files]
    )
    
    print("\n" + "="*50)
    print("🏆 FINAL PCNN HOLDOUT SCORES 🏆")
    print("="*50)
    print(f"Patient-Level AUROC: {metrics.auroc:.4f}")
    print(f"Lesion-Level AP:     {metrics.AP:.4f}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_dir", type=str, required=True, help="Path to nnUNet validation output folder")
    parser.add_argument("--gt_dir", type=str, required=True, help="Path to gt_segmentations folder in nnUNet_preprocessed")
    args = parser.parse_args()
    
    evaluate_metrics(Path(args.val_dir), Path(args.gt_dir))
