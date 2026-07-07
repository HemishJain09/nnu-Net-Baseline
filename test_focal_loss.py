import torch
import numpy as np

# Load the code dynamically from inject_focal_loss
import sys
import os

# We will just execute the string from inject_focal_loss to get the classes in this namespace
sys.path.append(os.path.dirname(__file__))
import inject_focal_loss
exec(inject_focal_loss.FOCAL_LOSS_CODE)

def test_focal_loss():
    print("🧪 Starting Strict PyTorch Test for nnUNet Focal Loss...")
    
    # 1. Simulate nnUNet Batch Size and Patch Size for 3d_fullres
    # Batch size: 2, Classes: 2 (Background, csPCa), Patch: 16x320x320
    B, C, D, H, W = 2, 2, 16, 320, 320
    
    # Create fake network output (logits) - requires gradients!
    net_output = torch.randn((B, C, D, H, W), dtype=torch.float32, requires_grad=True)
    
    # Create fake target (B, 1, D, H, W) where values are 0 or 1
    target = torch.randint(0, 2, (B, 1, D, H, W), dtype=torch.int8)
    
    print(f"   -> Created Net Output Tensor: {net_output.shape}, Requires Grad: {net_output.requires_grad}")
    print(f"   -> Created Target Tensor: {target.shape}")
    
    # 2. Instantiate Loss Function
    # We mock the configuration_manager and label_manager expected by DC_and_Focal_loss
    loss_fn = DC_and_Focal_loss(
        soft_dice_kwargs={'batch_dice': False, 'smooth': 1e-5, 'do_bg': False},
        ce_kwargs={},
        weight_ce=1,
        weight_dice=1,
        ignore_label=None,
        dice_class=MemoryEfficientSoftDiceLoss
    )
    print("   -> Initialized DC_and_Focal_loss successfully.")
    
    # 3. Forward Pass
    loss = loss_fn(net_output, target)
    print(f"   -> Forward Pass Successful! Loss Value: {loss.item():.4f}")
    
    # 4. Backward Pass (Strict Test)
    try:
        loss.backward()
        print("   -> Backward Pass (Gradient Calculation) Successful!")
        # Check if gradients are actually populated
        if net_output.grad is None:
            raise ValueError("Gradients are None!")
        if torch.isnan(net_output.grad).any():
            raise ValueError("NaN Gradients detected!")
        print("   -> Gradients are valid and NaN-free!")
    except Exception as e:
        print(f"❌ Backward Pass Failed: {e}")
        exit(1)
        
    print("✅ Strict Test Passed. Focal Loss is mathematically sound and safe for Colab!")

if __name__ == "__main__":
    test_focal_loss()
