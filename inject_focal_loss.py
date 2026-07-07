import os

FOCAL_LOSS_CODE = """import torch
from torch import nn
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import softmax_helper_dim1

class AdaptiveFocalLoss(nn.Module):
    def __init__(self, base_gamma=4.0, ignore_index=-100, **kwargs):
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
        
        # 1. The Silence Protocol: If the batch has NO cancer, drastically reduce its gradient
        # so it doesn't cause catastrophic forgetting on healthy patients.
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

class DC_and_AdaptiveFocal_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss):
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = AdaptiveFocalLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label not implemented for one hot encoded targets'
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

class nnUNetTrainerFocalLoss(nnUNetTrainer):
    def build_loss(self):
        loss = DC_and_AdaptiveFocal_loss(
            {'batch_dice': self.configuration_manager.batch_dice,
             'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            {},
            weight_ce=2, weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss
        )
        return loss
"""

def inject():
    # Detect if we are in Colab by checking for /content/nnUNet
    nnunet_dir = "/content/nnUNet"
    if not os.path.exists(nnunet_dir):
        # Fallback to current environment for local testing
        import nnunetv2
        nnunet_dir = os.path.dirname(os.path.dirname(nnunetv2.__file__))
        
    trainer_dir = os.path.join(nnunet_dir, "nnunetv2", "training", "nnUNetTrainer")
    if not os.path.exists(trainer_dir):
        raise FileNotFoundError(f"Could not find nnUNetTrainer directory at {trainer_dir}")
        
    target_file = os.path.join(trainer_dir, "nnUNetTrainerFocalLoss.py")
    with open(target_file, "w") as f:
        f.write(FOCAL_LOSS_CODE)
        
    print(f"✅ Successfully injected Adaptive nnUNetTrainerFocalLoss into {target_file}")

if __name__ == "__main__":
    inject()
