import torch
import blosc2
import numpy as np
import os
import glob
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

def check_probs():
    print("="*50)
    print("🧠 DIAGNOSING NETWORK PROBABILITIES")
    print("="*50)
    
    # 1. Initialize Predictor
    print("\n[1/3] Loading nnU-Net Predictor & Weights...")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

    checkpoint_dir = '/content/drive/MyDrive/PI-CAI_nnUNet_Results/Dataset500_PICAI/nnUNetTrainerFocalLoss__nnUNetPlans__3d_fullres'
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Error: Checkpoint directory not found at {checkpoint_dir}")
        print("Please ensure your Google Drive is mounted!")
        return

    try:
        predictor.initialize_from_trained_model_folder(
            checkpoint_dir,
            use_folds=(0,),
            checkpoint_name='checkpoint_latest.pth',
        )
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return
    
    # 2. Get a sample from the cache
    print("\n[2/3] Extracting a 3D MRI from Blosc2 Cache...")
    cache_dir = '/content/nnUNet_preprocessed/Dataset500_PICAI/nnUNetPlans_3d_fullres'
    b2nd_masks = sorted(glob.glob(f'{cache_dir}/*_seg.b2nd'))
    
    if len(b2nd_masks) == 0:
        print("❌ No cache files found! Ensure you are running this in Colab.")
        return
        
    sample_mask = b2nd_masks[-1] # Pick the last one
    sample_data = sample_mask.replace('_seg.b2nd', '.b2nd')
    
    print(f"      Loading: {os.path.basename(sample_data)}")
    data = blosc2.open(urlpath=sample_data, mode='r')[:]
    
    # The network is mathematically built for 16x320x320 sliding window patches.
    # To avoid U-Net decoder shape mismatch, we will extract a perfect central crop.
    Z, Y, X = data.shape[1:]
    target_z, target_y, target_x = 16, 320, 320
    
    start_z = max(0, (Z - target_z) // 2)
    start_y = max(0, (Y - target_y) // 2)
    start_x = max(0, (X - target_x) // 2)
    
    crop = np.zeros((data.shape[0], target_z, target_y, target_x), dtype=data.dtype)
    sz = min(target_z, Z - start_z)
    sy = min(target_y, Y - start_y)
    sx = min(target_x, X - start_x)
    
    crop[:, :sz, :sy, :sx] = data[:, start_z:start_z+sz, start_y:start_y+sy, start_x:start_x+sx]
    
    # Convert to tensor (1, 6, 16, 320, 320)
    tensor_data = torch.from_numpy(crop).unsqueeze(0).cuda()
    
    print("\n[3/3] Running Forward Pass (Full 3D Volume)...")
    predictor.network.cuda()
    predictor.network.eval()
    with torch.no_grad():
        logits = predictor.network(tensor_data)
        if isinstance(logits, list):
            logits = logits[0]
            
        # Convert logits to probabilities via softmax
        probs = torch.softmax(logits, dim=1)
        # Extract the probability map for class 1 (Cancer)
        fg_probs = probs[0, 1, :, :, :] 
        
        max_prob = fg_probs.max().item()
        mean_prob = fg_probs.mean().item()
        median_prob = torch.median(fg_probs).item()
        
        print("\n" + "="*50)
        print("📊 PROBABILITY ANALYSIS (CANCER CLASS)")
        print("="*50)
        print(f"Max Probability anywhere in volume:  {max_prob:.6f}")
        print(f"Mean Probability across volume:    {mean_prob:.6f}")
        print(f"Median Probability across volume:  {median_prob:.6f}")
        print("="*50)
        
        if max_prob < 0.5:
            print("\n🚨 DIAGNOSIS: The Zero-Dice Trap!")
            print("The model is learning, but it is currently too scared to push any probability above 0.50.")
            print("Because the highest probability is < 0.50, the argmax mathematically rounds everything to 0.")
            print("This perfectly explains why your Dice score is exactly 0.0.")
            print("SOLUTION: Run the injected Focal Loss update to massively penalize False Negatives, then resume training.")
        else:
            print("\n✅ BREAKOUT DETECTED!")
            print("The model is successfully pushing probabilities > 0.50!")
            print("You should start seeing positive Dice scores now.")
            
if __name__ == '__main__':
    check_probs()
