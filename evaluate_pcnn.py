import os
import shutil
import subprocess
import argparse
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path

def setup_pcnn_test_set(nnunet_raw: Path, marksheet_path: Path):
    print("--- Setting up PCNN Holdout Set ---")
    dataset_dir = nnunet_raw / "Dataset500_PICAI"
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    
    images_ts = dataset_dir / "imagesTs_PCNN"
    labels_ts = dataset_dir / "labelsTs_PCNN"
    
    if images_ts.exists():
        shutil.rmtree(images_ts)
    if labels_ts.exists():
        shutil.rmtree(labels_ts)
        
    images_ts.mkdir(parents=True)
    labels_ts.mkdir(parents=True)
    
    df = pd.read_csv(marksheet_path)
    df['patient_id'] = df['patient_id'].astype(str)
    
    # Get PCNN patient IDs
    pcnn_patients = df[df['center'] == 'PCNN']['patient_id'].tolist()
    
    # We also need to map patient_id to case_id (patient_id + study_id)
    # The files in imagesTr are formatted as case_id_0000.nii.gz
    # So we search the directory
    all_label_files = list(labels_tr.glob("*.nii.gz"))
    pcnn_cases = []
    
    print(f"DEBUG: Found {len(all_label_files)} total files in {labels_tr}")
    if len(all_label_files) > 0:
        print(f"DEBUG: First file name: {all_label_files[0].name}")
    print(f"DEBUG: Total PCNN patients in marksheet: {len(pcnn_patients)}")
    if len(pcnn_patients) > 0:
        print(f"DEBUG: First 5 PCNN patient IDs from marksheet: {pcnn_patients[:5]}")
        
    for label_file in all_label_files:
        case_id = label_file.name.replace(".nii.gz", "")
        patient_id = case_id.split("_")[0]
        if patient_id in pcnn_patients:
            pcnn_cases.append(case_id)
            # Symlink label
            os.symlink(label_file, labels_ts / label_file.name)
            
            # Symlink 6 channels
            for i in range(6):
                img_file = images_tr / f"{case_id}_{i:04d}.nii.gz"
                if img_file.exists():
                    os.symlink(img_file, images_ts / f"{case_id}_{i:04d}.nii.gz")
                    
    print(f"Successfully isolated {len(pcnn_cases)} PCNN cases for inference.")
    return images_ts, labels_ts, pcnn_cases

def run_inference(images_ts: Path, output_dir: Path):
    print("\n--- Running nnUNetv2 Inference ---")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "nnUNetv2_predict",
        "-i", str(images_ts),
        "-o", str(output_dir),
        "-d", "500",
        "-c", "3d_fullres",
        "-f", "0",
        "-tr", "nnUNetTrainerFocalLoss",
        "--save_probabilities" # Essential for AUROC evaluation
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("Inference completed.")

def convert_npz_to_nifti_probabilities(output_dir: Path, pcnn_cases: list, labels_ts: Path):
    """
    nnU-Net exports probabilities as .npz files. 
    picai_eval requires them as NIfTI images for spatial alignment.
    This function unpacks the .npz and saves the cancer class (class 1) as a .nii.gz
    """
    print("\n--- Converting Probabilities to NIfTI format for picai_eval ---")
    prob_dir = output_dir / "probabilities_nii"
    prob_dir.mkdir(parents=True, exist_ok=True)
    
    for case_id in pcnn_cases:
        npz_file = output_dir / f"{case_id}.npz"
        pkl_file = output_dir / f"{case_id}.pkl"
        
        if not npz_file.exists():
            print(f"Warning: {npz_file} not found. Skipping...")
            continue
            
        # Read the ground truth to get the exact original geometry
        ref_img = sitk.ReadImage(str(labels_ts / f"{case_id}.nii.gz"))
        
        # Load the npz probability array
        data = np.load(npz_file)
        probs = data['probabilities'] # shape: (num_classes, Z, Y, X)
        
        # We want the probability of Class 1 (Cancer)
        cancer_prob = probs[1]
        
        # Convert back to SimpleITK image
        prob_img = sitk.GetImageFromArray(cancer_prob)
        prob_img.CopyInformation(ref_img)
        
        sitk.WriteImage(prob_img, str(prob_dir / f"{case_id}.nii.gz"))
        
    return prob_dir

def evaluate_metrics(prob_dir: Path, labels_ts: Path):
    print("\n--- Running Official picai_eval Metrics ---")
    try:
        from picai_eval import evaluate
        from reportless_metrics import Report
    except ImportError:
        print("picai_eval not installed. Please run: pip install picai_eval")
        return
        
    # Gather files
    y_true_files = sorted(list(labels_ts.glob("*.nii.gz")))
    y_det_files = sorted(list(prob_dir.glob("*.nii.gz")))
    
    if len(y_true_files) != len(y_det_files):
        print("Mismatch between ground truth and predictions!")
        return
        
    y_true_files = [str(f) for f in y_true_files]
    y_det_files = [str(f) for f in y_det_files]
    
    # Run PI-CAI Evaluation
    metrics = evaluate(
        y_true=y_true_files,
        y_det=y_det_files,
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
    parser.add_argument("--nnunet_raw", type=str, required=True)
    parser.add_argument("--marksheet", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    images_ts, labels_ts, pcnn_cases = setup_pcnn_test_set(Path(args.nnunet_raw), Path(args.marksheet))
    
    if not pcnn_cases:
        print("No PCNN cases found. Exiting.")
        exit(1)
        
    run_inference(images_ts, Path(args.output_dir))
    
    prob_dir = convert_npz_to_nifti_probabilities(Path(args.output_dir), pcnn_cases, labels_ts)
    
    evaluate_metrics(prob_dir, labels_ts)
