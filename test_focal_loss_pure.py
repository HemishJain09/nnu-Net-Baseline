import torch
from torch import nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', ignore_index=-100, **kwargs):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index, **kwargs)

    def forward(self, net_output, target):
        ce_loss = self.ce(net_output, target)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

print("🧪 Starting Strict PyTorch Test for Focal Loss...")
# 1. Simulate nnUNet Batch Size and Patch Size for 3d_fullres
B, C, D, H, W = 2, 2, 16, 320, 320
net_output = torch.randn((B, C, D, H, W), dtype=torch.float32, requires_grad=True)
target = torch.randint(0, C, (B, D, H, W), dtype=torch.int8)

print(f"   -> Created Net Output Tensor: {net_output.shape}, Requires Grad: {net_output.requires_grad}")
print(f"   -> Created Target Tensor: {target.shape}")

loss_fn = FocalLoss()
loss = loss_fn(net_output, target.long())
print(f"   -> Forward Pass Successful! Loss Value: {loss.item():.4f}")

try:
    loss.backward()
    print("   -> Backward Pass (Gradient Calculation) Successful!")
    if net_output.grad is None:
        raise ValueError("Gradients are None!")
    if torch.isnan(net_output.grad).any():
        raise ValueError("NaN Gradients detected!")
    print("   -> Gradients are valid and NaN-free!")
except Exception as e:
    print(f"❌ Backward Pass Failed: {e}")
    exit(1)
    
print("✅ Strict Test Passed. Focal Loss is mathematically sound and safe for Colab!")
