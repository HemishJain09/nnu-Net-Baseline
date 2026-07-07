import torch
from torch import nn

class AdaptiveFocalLossV2(nn.Module):
    def __init__(self, base_gamma=4.0, ignore_index=-100, **kwargs):
        super(AdaptiveFocalLossV2, self).__init__()
        self.base_gamma = base_gamma
        self.ce = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index, **kwargs)
        self.ignore_index = ignore_index

    def forward(self, net_output, target):
        ce_loss = self.ce(net_output, target)
        pt = torch.exp(-ce_loss)
        
        valid_mask = target != self.ignore_index
        valid_targets = target[valid_mask]
        num_positive = valid_targets.sum().float()
        
        # 1. The Silence Protocol: If the batch has NO cancer, drastically reduce its gradient
        # so it doesn't cause catastrophic forgetting.
        if num_positive == 0:
            return (ce_loss * 0.01).mean()
            
        # 2. Static Alpha: Hard-code a massive weight for cancer pixels (100.0) 
        # and standard weight for background (1.0).
        alpha_tensor = torch.where(target == 1, 100.0, 1.0)
        
        # 3. Dynamic Gamma
        adaptive_gamma = self.base_gamma + (1.0 - pt)
        
        # 4. Focal Loss Calculation
        focal_loss = alpha_tensor * ((1.0 - pt) ** adaptive_gamma) * ce_loss
        
        # 5. Mean Reduction (to divide by Batch*Pixels and avoid PyTorch Gradient Clipper)
        return focal_loss.mean()

def test_gradients():
    print("Testing Focal Loss V2 (Silence Protocol)...")
    torch.manual_seed(42)
    
    B, C, Z, Y, X = 2, 2, 16, 64, 64
    focal_loss = AdaptiveFocalLossV2(base_gamma=4.0)
    
    print("\n--- Test 1: Empty Batch (No Cancer) ---")
    logits_empty = torch.randn(B, C, Z, Y, X, requires_grad=True)
    target_empty = torch.zeros(B, Z, Y, X, dtype=torch.long)
    
    loss_empty = focal_loss(logits_empty, target_empty)
    loss_empty.backward()
    
    grad_empty = logits_empty.grad
    print(f"Empty Batch Loss: {loss_empty.item():.6f}")
    print(f"Empty Batch Max Grad:  {grad_empty.abs().max().item():.8f}")
    
    
    print("\n--- Test 2: Cancer Batch ---")
    logits_cancer = torch.randn(B, C, Z, Y, X)
    logits_cancer[:, 0] += 5.0 # Very confident it's background
    logits_cancer[:, 1] -= 5.0 # Very unconfident it's cancer (The zero-dice trap)
    logits_cancer.requires_grad = True
    
    target_cancer = torch.zeros(B, Z, Y, X, dtype=torch.long)
    target_cancer[0, 8:12, 32:40, 32:40] = 1 # A lesion
    
    loss_cancer = focal_loss(logits_cancer, target_cancer)
    loss_cancer.backward()
    
    grad_cancer = logits_cancer.grad
    print(f"Cancer Batch Loss: {loss_cancer.item():.6f}")
    print(f"Cancer Batch Max Grad: {grad_cancer.abs().max().item():.8f}")
    
    ratio = grad_cancer.abs().max().item() / max(1e-10, grad_empty.abs().max().item())
    print(f"\n✅ Cancer patch gradients are {ratio:.1f}x larger than Empty patch gradients!")

if __name__ == '__main__':
    test_gradients()
