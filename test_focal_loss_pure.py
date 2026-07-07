import torch
from torch import nn

class AdaptiveFocalLoss(nn.Module):
    def __init__(self, base_gamma=2.0, ignore_index=-100, **kwargs):
        super(AdaptiveFocalLoss, self).__init__()
        self.base_gamma = base_gamma
        self.ce = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index, **kwargs)
        self.ignore_index = ignore_index

    def forward(self, net_output, target):
        ce_loss = self.ce(net_output, target)
        pt = torch.exp(-ce_loss)
        
        valid_mask = target != self.ignore_index
        valid_targets = target[valid_mask]
        
        num_positive = valid_targets.sum().float()
        total_valid = valid_mask.sum().float()
        
        if total_valid == 0 or num_positive == 0 or num_positive == total_valid:
            alpha_tensor = torch.ones_like(target, dtype=torch.float32)
        else:
            pos_ratio = num_positive / total_valid
            neg_ratio = 1.0 - pos_ratio
            alpha_tensor = torch.where(target == 1, neg_ratio, pos_ratio)
            
        adaptive_gamma = self.base_gamma + (1.0 - pt)
        focal_loss = alpha_tensor * ((1.0 - pt) ** adaptive_gamma) * ce_loss
        
        # Normalize by the sum of weights to preserve gradient magnitude
        return focal_loss.sum() / alpha_tensor.sum()

print("🧪 Starting Strict PyTorch Test for ADAPTIVE Focal Loss...")

def run_test(name, target_tensor):
    B, C, D, H, W = 2, 2, 16, 320, 320
    net_output = torch.randn((B, C, D, H, W), dtype=torch.float32, requires_grad=True)
    
    loss_fn = AdaptiveFocalLoss()
    try:
        loss = loss_fn(net_output, target_tensor.long())
        loss.backward()
        if net_output.grad is None or torch.isnan(net_output.grad).any():
            raise ValueError("NaN or None Gradients detected!")
        print(f"   ✅ {name} Passed! Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"   ❌ {name} Failed! Error: {e}")
        exit(1)

B, D, H, W = 2, 16, 320, 320

target_50 = torch.randint(0, 2, (B, D, H, W), dtype=torch.int8)
run_test("Test 1: 50% Cancer Batch", target_50)

target_0 = torch.zeros((B, D, H, W), dtype=torch.int8)
run_test("Test 2: 0% Cancer Batch (Benign)", target_0)

target_100 = torch.ones((B, D, H, W), dtype=torch.int8)
run_test("Test 3: 100% Cancer Batch", target_100)

target_1 = torch.zeros((B, D, H, W), dtype=torch.int8)
num_elements = target_1.numel()
target_1.view(-1)[:int(num_elements * 0.01)] = 1
run_test("Test 4: 1% Cancer Batch (Extreme Imbalance)", target_1)
