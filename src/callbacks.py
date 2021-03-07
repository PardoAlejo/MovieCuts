from pytorch_lightning import Callback
import torch
import numpy as np
import os

def sigmoid(X):
    return 1/(1+torch.exp(-X.squeeze()))

class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)

class ValidationBatchAccumulator(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        logits, labels = outputs
        pl_module.val_logits_epoch.extend(logits)
        pl_module.val_labels_epoch.extend(labels)

class APClassAnalysis(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()

    def on_validation_epoch_end(self, trainer, pl_module):
        all_logits = torch.stack(pl_module.val_logits_epoch,dim=0)
        all_labels = torch.stack(pl_module.val_labels_epoch,dim=0)
        cnfmat = pl_module.confusion_matrix(sigmoid(all_logits), all_labels).cpu().numpy()
        f1_per_class = pl_module.f1_per_class(sigmoid(all_logits), all_labels).cpu().numpy()
        cut_types = pl_module.cut_types
        save_dir = f'{trainer.log_dir}/{trainer.logger.name}/class_metrics'
        if not os.path.exists(f'{save_dir}'):
            os.makedirs(save_dir)
        np.savetxt(f'{save_dir}/f1_per_class-epoch_{trainer.current_epoch}.txt', f1_per_class, header=str(cut_types), footer=f'Mean: {str(f1_per_class.mean())}')
        np.savetxt(f'{save_dir}/confustion_matrix_{trainer.current_epoch}.txt', cnfmat, header=str(cut_types))