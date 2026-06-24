
import math
import warnings
from torch.optim.lr_scheduler import _LRScheduler


class MyLR(_LRScheduler):
    def __init__(self, optimizer, T_max, phase1_epoch, eta_min, last_epoch=-1, verbose=False, decay=1):
        self.T_max = T_max
        self.eta_min = eta_min  # minimum learning rate
        self.phase1_epoch = phase1_epoch  # epoch index where phase 1 ends
        self.decay = decay
        # Do not override self.base_lrs; the parent class reads initial LRs from the optimizer
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        
        # Initial step (last_epoch=0): return the optimizer's current learning rates
        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        
        # Phase 1 (last_epoch < phase1_epoch): cosine annealing
        elif self.last_epoch < self.phase1_epoch:
            return [
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) /
                (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups
            ]
        
        # Phase 2 (last_epoch >= phase1_epoch): hold at eta_min
        else:
            return [self.eta_min for _ in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        # Closed-form LR for logging; uses parent-provided self.base_lrs list
        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
            for base_lr in self.base_lrs
        ]
