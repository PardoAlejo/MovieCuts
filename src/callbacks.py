from typing import Any, List, Optional, Union
from pytorch_lightning import Callback
import torch
import numpy as np
import os
from torch.nn import functional as F
from pytorch_lightning.metrics import Metric
from pytorch_lightning import metrics
from pytorch_lightning.metrics.functional.average_precision import _average_precision_compute, _average_precision_update
import pandas as pd
import pickle

def sigmoid(X):
    return 1/(1+torch.exp(-X.squeeze()))


class MultilabelAP(Metric):
    def __init__(
        self,
        num_classes: Optional[int] = None,
        pos_label: Optional[int] = None,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )
        "Returns a list with two elements: AP per class, and mAP"
        if num_classes <= 1:
            raise ValueError(
                f'Argument `num_classes` was set to {num_classes} in'
                f' metric `MultilabelAP` num_classes has to be > 1'
            )

        self.num_classes = num_classes
        self.pos_label = pos_label

        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("target", default=[], dist_reduce_fx=None)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target, num_classes, pos_label = _average_precision_update(
            preds, target, self.num_classes, self.pos_label
        )
        self.preds.append(preds)
        self.target.append(target.reshape(-1,10))
        self.num_classes = num_classes
        self.pos_label = pos_label

    def compute(self) -> Union[torch.Tensor, List[torch.Tensor]]:
        preds = torch.cat(self.preds, dim=0)
        target = torch.cat(self.target, dim=0)
        
        # Duplicate instances with multilabels
        all_preds = []
        all_targets = []
        for i in range(preds.shape[0]):
            this_targets = torch.nonzero(target[i])
            for label in this_targets:
                all_preds.append(preds[i])
                all_targets.append(label)

        all_preds = torch.stack(all_preds)
        all_targets = torch.cat(all_targets)
        
        ap_per_class = _average_precision_compute(all_preds, all_targets, self.num_classes, self.pos_label)
        mAP = torch.stack([x for x in ap_per_class if not torch.isnan(x)]).mean()
        return [mAP, torch.stack(ap_per_class)]


class WriteMetricReport(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
    
    def on_validation_epoch_end(self, trainer, pl_module):
        cnfmat = pl_module.confusion_matrix_val.compute().cpu().numpy()
        f1_per_class = pl_module.f1_per_class_val.compute().cpu().numpy()
        mAP, ap_per_class = pl_module.ap_per_class_val.compute()

        #Prepare data to save it
        accuracys = cnfmat.diagonal().tolist(); accuracys.insert(0,cnfmat.diagonal().mean())
        accuracys.insert(0,'Acc')
        f1s = f1_per_class.tolist(); f1s.insert(0,f1_per_class.mean())
        f1s.insert(0,'f1')
        aps = ap_per_class.cpu().numpy().tolist(); aps.insert(0,mAP.cpu().numpy())
        aps.insert(0,'AP')
        cut_types = pl_module.cut_types
        headers = ['Metric','Mean']+cut_types
        metrics_df = pd.DataFrame([accuracys,f1s,aps], columns=headers)
        save_dir = f'{trainer.log_dir}/{trainer.logger.name}/class_metrics'
        if not os.path.exists(f'{save_dir}'):
            os.makedirs(save_dir)
        metrics_df.to_csv(f'{save_dir}/metrics-epoch_{trainer.current_epoch}.csv')
        np.savetxt(f'{save_dir}/confustion_matrix_{trainer.current_epoch}.txt', cnfmat)

    def on_test_epoch_end(self, trainer, pl_module):
        if pl_module.args.finetune_validation:
            pass
        else:
            cnfmat = pl_module.confusion_matrix_val.compute().cpu().numpy()
            f1_per_class = pl_module.f1_per_class_val.compute().cpu().numpy()
            mAP, ap_per_class = pl_module.ap_per_class_val.compute()

            #Prepare data to save it
            accuracys = cnfmat.diagonal().tolist(); accuracys.insert(0,cnfmat.diagonal().mean())
            accuracys.insert(0,'Acc')
            f1s = f1_per_class.tolist(); f1s.insert(0,f1_per_class.mean())
            f1s.insert(0,'f1')
            aps = ap_per_class.cpu().numpy().tolist(); aps.insert(0,mAP.cpu().numpy())
            aps.insert(0,'AP')
            cut_types = pl_module.cut_types
            headers = ['Metric','Mean']+cut_types
            metrics_df = pd.DataFrame([accuracys,f1s,aps], columns=headers)
            save_dir = f'{trainer.log_dir}/{trainer.logger.name}/class_metrics_test'
            if not os.path.exists(f'{save_dir}'):
                os.makedirs(save_dir)
            metrics_df.to_csv(f'{save_dir}/metrics-epoch_{trainer.current_epoch}.csv')
            np.savetxt(f'{save_dir}/confustion_matrix_{trainer.current_epoch}.txt', cnfmat)

class SaveLogits(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
    
    def on_test_end(self, trainer, pl_module):
        split = 'test' if pl_module.args.finetune_test else 'val'
        all_logits = torch.cat(pl_module.inference_logits_epoch).detach().cpu().numpy()
        clip_names = [name for batch in pl_module.clip_names_epoch for name in batch]
        logits = dict(zip(clip_names, all_logits))
        save_dir = f'{trainer.log_dir}'
        if not os.path.exists(f'{save_dir}'):
                os.makedirs(save_dir)
        with open(f'{save_dir}/{split}_logits.pkl', 'wb') as f:
            pickle.dump(logits,f)