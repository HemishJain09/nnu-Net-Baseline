# PI-CAI nnUNet v2 Pipeline — Architecture & Documentation

This document provides a comprehensive overview of the domain-adaptive PI-CAI nnU-Net v2 pipeline. It explains the high-level architecture, what each Python script does, and how the Google Colab environment executes the training loop.

---

## 1. High-Level Architecture

The goal of this pipeline is to take the raw PI-CAI dataset, convert it into nnU-Net's strict formatting requirements, and train a highly optimized 3D image segmentation model. 

### Data Flow
1. **Google Drive (Source)**: Contains the original PI-CAI dataset (`t2`, `adc`, etc.) and acts as persistent storage for the trained model checkpoints.
2. **Local SSD (Compute)**: Because Google Drive is too slow for GPU data loading, the pipeline copies all data to the Colab instance's ultra-fast local SSD (`/content/`). 
3. **Preprocessed Cache System**: To protect against Colab runtime disconnections, the pipeline automatically compresses the fully prepared data into a Zip Cache (`PI-CAI_nnUNet_Preprocessed_Cache.zip`) and saves it to Drive. Upon resume, it unzips this in 5 minutes, skipping 2.5 hours of manual data preparation.

---

## 2. File-by-File Breakdown

### `convert_picai_to_nnunet.py`
**Purpose**: Transforms the PI-CAI folder structure into the `nnUNet_raw` format required by nnU-Net v2.
- **6-Channel Input**: It stacks `T2W`, `ADC`, and `HBV` alongside 3 newly generated channels (One-hot encoded Zonal Background, Peripheral Zone, and Transition Zone).
- **Function `one_hot_encode_zonal_mask`**: Reads the PI-CAI Zonal Mask (where 0=BG, 1=PZ, 2=TZ) and splits it into 3 separate binary `.nii.gz` channels.
- **Function `generate_empty_mask`**: The PI-CAI dataset has 207 "benign" patients who do not have lesion masks. nnU-Net crashes if a mask is missing. This function detects missing masks and generates a blank (all-zeros) NIfTI file matching the patient's exact geometry.
- **Function `copy_and_binarize_mask`**: Some PI-CAI lesion masks contain values like `3` or `5`. nnU-Net strictly requires a binary `[0, 1]` mask for csPCa. This function reads the mask, forces all values > 0 to `1`, and saves it.

### `generate_splits.py`
**Purpose**: Handles **Domain Adaptation (Pure Holdout)**. Instead of randomly splitting the dataset, it controls exactly which hospitals the neural network is allowed to see.
- **Logic**: 
  1. Reads `marksheet.csv` using Pandas.
  2. Extracts the `patient_id` from the image filenames and matches it to the clinical `center` (RUMC, ZGT, or PCNN).
  3. Uses the `--train_centers` argument to **keep** patients from designated hospitals (e.g., RUMC, ZGT) and **completely discard** patients from the holdout hospital (PCNN).
  4. It then uses Scikit-Learn's `GroupKFold` to create 5 cross-validation folds strictly out of the kept centers. 
  5. Outputs `splits_final.json`, which nnU-Net reads before training.

### `nnunet_v2_picai_colab.ipynb`
**Purpose**: The central orchestrator. This notebook automates the entire process in Google Colab.
- **Phase 1 (Sanity Check)**: Runs the whole pipeline on just 10 patients for 5 epochs. This allows you to verify that everything works end-to-end without spending hours of compute.
- **Phase 2 (Production)**: The real 1,500-patient pipeline. It executes the scripts, manages the Zip Cache, and modifies the underlying nnU-Net source code (`sed -i 's/self.num_epochs = 1000/self.num_epochs = 250/g'`) to optimize training time before launching the GPU training loop.

---

## 3. The Step-by-Step Execution Pipeline

When you run Phase 2 in Colab, here is exactly what happens under the hood:

> [!NOTE]
> **Step 1: Cleanup & Setup**
> The notebook wipes any leftover sanity-check data from the local SSD to ensure a clean slate. It clones the official `MIC-DKFZ/nnUNet` repository and installs it in "editable" mode so we can hack the source code later.

> [!TIP]
> **Step 2 & 3: Zip Cache Auto-Resume**
> The notebook looks for `PI-CAI_nnUNet_Preprocessed_Cache.zip` on your Google Drive. 
> - **Cache Miss**: It executes `convert_picai_to_nnunet.py` to convert all 1,500 patients. It then runs `nnUNetv2_plan_and_preprocess` (which extracts dataset fingerprints and normalizes intensities). Finally, it zips the output and saves it to Drive.
> - **Cache Hit**: If Colab disconnected previously, it just unzips the cached file to the SSD in 5 minutes, instantly restoring your state.

> [!IMPORTANT]
> **Step 4: Domain Splitting**
> `generate_splits.py` is executed, passing `--train_centers RUMC ZGT`. The 3rd center (PCNN) is quietly scrubbed from the `splits_final.json` file. The neural network will now be completely blind to PCNN.

> [!CAUTION]
> **Step 5: Training Loop**
> The notebook uses a bash `sed` command to crack open the internal `nnUNetTrainer.py` source file and forcefully change `self.num_epochs` from 1000 to 250. 
> Finally, `!nnUNetv2_train 500 3d_fullres 0` is called. The GPU spins up, reads from the local SSD, and begins training Fold 0. After every 50 epochs, the model state is saved securely into your persistent Google Drive folder.

---

## 4. nnU-Net Internal Preprocessing Explained

When `nnUNetv2_plan_and_preprocess` runs, it executes highly optimized, autonomous operations on your data. Here is exactly what happens under the hood, and why it is safe for our 6-channel setup:

### A. Intensity Normalization
By default, nnU-Net calculates the mean and standard deviation of intensities across the dataset and applies **Z-Score Normalization** `(x - mean) / std` to the T2, ADC, and HBV channels. 
**What about the Zonal Masks?** Because the 3 Zonal Masks are provided as input channels (Channels 3, 4, and 5), nnU-Net treats them like MRI scans and will also apply Z-score normalization. This simply shifts the binary `0` and `1` values to something like `-0.5` and `+2.3`. This **will not break the pipeline**; convolutional neural networks are completely scale-invariant and will effortlessly learn this new continuous distribution.

### B. Resampling (Voxel Spacing)
MRI scans from different hospitals often have different voxel dimensions (e.g., $3mm \times 0.5mm \times 0.5mm$). nnU-Net calculates the median spacing of the dataset and resamples all images to match. 
**Interpolation**: It uses 3rd-order spline interpolation for the input channels, and Nearest Neighbor for the lesion mask. This means your Zonal Masks will get slightly "smoothed" at the edges (producing soft, continuous boundaries instead of harsh binary pixels). This is actually beneficial for spatial priors as it reduces aliasing artifacts.

### C. Dynamic Data Augmentation (During Training)
While training, the dataloader applies aggressive "on-the-fly" data augmentation. It will randomly rotate, scale, flip, and apply elastic deformations to your images to artificially expand your dataset and prevent overfitting. It seamlessly applies these identical transformations to all 6 input channels and the lesion mask simultaneously, ensuring perfect spatial alignment is never broken.
