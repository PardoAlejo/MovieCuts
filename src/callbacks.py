from typing import Any, List, Optional, Union
from pytorch_lightning import Callback
import torch
import numpy as np
import json, logging, glob, pathlib, pickle, os, sys
import os.path as osp
sys.path.insert(1, f'{os.getcwd()}/utils')
from wandb_utils import Wandb
from config import config
from torch.nn import functional as F
import torch.distributed as dist
from pytorch_lightning.metrics import Metric
from pytorch_lightning import metrics
import pandas as pd
from sklearn.metrics import average_precision_score

def sigmoid(X):
    return 1/(1+torch.exp(-X.squeeze()))

def get_experiment_version(config):
    versions = glob.glob(f'{os.getcwd()}/{config.exp_dir}/{config.exp_name}/version_*')
    logging.info(f"Experiment Version: {len(versions)}")
    return len(versions)

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
        if preds.ndim == 1:
            preds = preds.unsqueeze(0)
        if target.ndim == 1:
            target = target.unsqueeze(0)
            
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> Union[torch.Tensor, List[torch.Tensor]]:
        preds = torch.cat(self.preds, dim=0)
        target = torch.cat(self.target, dim=0)
        ap_per_class = []
        for i in range(self.num_classes): 
            this_pred = preds[:,i].detach().cpu() 
            this_target = target[:,i].detach().cpu() 
            this_ap = average_precision_score(this_target, this_pred) 
            ap_per_class.append(this_ap)
            
        mAP = np.mean(ap_per_class)
        return [mAP, ap_per_class]


class WriteMetricReport(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
    
    def on_validation_epoch_end(self, trainer, pl_module):
        mAP, ap_per_class = pl_module.ap_per_class_val.compute()
        # print(f'AP per class len of targets : {len(pl_module.ap_per_class_val.target[0])}')
        # print(f'AP per class len of logits : {len(pl_module.ap_per_class_val.preds[-1])}')
        #Prepare data to save it
        aps = ap_per_class; aps.insert(0,mAP)
        aps.insert(0,'AP')
        cut_types = pl_module.cut_types
        headers = ['Metric','Mean']+cut_types
        print(f'This Model mAP: {aps[1]*100:.2f}%')
        for cl, apcl in zip(headers[2:], aps[2:]):
            print(f'AP for class {cl}: {apcl*100:.2f}%')
        metrics_df = pd.DataFrame([aps], columns=headers)
        save_dir = f'{trainer.log_dir}/{trainer.logger.name}/class_metrics'
        if not os.path.exists(f'{save_dir}'):
            os.makedirs(save_dir, exist_ok=True)
        metrics_df.to_csv(f'{save_dir}/metrics-epoch_{trainer.current_epoch}.csv')

    def on_test_epoch_end(self, trainer, pl_module):

        mAP, ap_per_class = pl_module.ap_per_class_test.compute()
        # print(f'AP per class len of targets : {len(pl_module.ap_per_class_test.target[0])}')
        # print(f'AP per class len of logits : {len(pl_module.ap_per_class_test.preds[-1])}')
        #Prepare data to save it
        aps = ap_per_class; aps.insert(0,mAP)
        aps.insert(0,'AP')
        cut_types = pl_module.cut_types
        headers = ['Metric','Mean']+cut_types
        metrics_df = pd.DataFrame([aps], columns=headers)
        print(f'This Model mAP: {aps[1]*100:.2f}%')
        for cl, apcl in zip(headers[2:], aps[2:]):
            print(f'AP for {cl}: {apcl*100:.2f}%')
        if pl_module.config.inference.test:
            save_dir = f'{trainer.log_dir}/class_metrics_test'
            if not os.path.exists(f'{save_dir}'):
                os.makedirs(save_dir, exist_ok=True)
            metrics_df.to_csv(f'{save_dir}/metrics-epoch_{trainer.current_epoch}.csv')
        elif pl_module.config.inference.validation:
            save_dir = f'{trainer.log_dir}/class_metrics_val'
            if not os.path.exists(f'{save_dir}'):
                os.makedirs(save_dir, exist_ok=True)
            metrics_df.to_csv(f'{save_dir}/metrics-epoch_{trainer.current_epoch}.csv')

class SaveLogits(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
    
    def on_test_epoch_end(self, trainer, pl_module):
        split = 'test' if pl_module.config.inference.test else 'val'
        # print(f'Len of inf logits {len(pl_module.inference_logits_epoch[0])}')
        # print(f'Len of inf logits {len(pl_module.inference_logits_epoch[-1])}')
        all_logits = torch.cat(pl_module.inference_logits_epoch).detach().cpu().numpy()
        clip_names = [name for batch in pl_module.clip_names_epoch for name in batch]
        logits = dict(zip(clip_names, all_logits))
        save_dir = f'{trainer.logger.log_dir}'
        if not os.path.exists(f'{save_dir}'):
                os.makedirs(save_dir, exist_ok=True)
        with open(f'{save_dir}/{split}_logits.pkl', 'wb') as f:
            pickle.dump(logits,f)

class wandb_config(Callback):
    """Initialize WANDB and Config options"""
    def __init__(self, opt, config):
        self.config = config
        self.opt = opt
        super().__init__()
        
    def on_init_start(self, trainer):
        
        if trainer.is_global_zero:
            config = self.config
            version_num = get_experiment_version(config)
            logging.info(osp.join(config.exp_dir, config.exp_name, f'version_{version_num}'))
            config.log_dir = osp.join(config.exp_dir, config.exp_name, f'version_{version_num}')
            config.ckpt_dir = os.path.join(config.log_dir, 'checkpoints')
            config.code_dir = os.path.join(config.log_dir, 'code')
            pathlib.Path(config.log_dir).mkdir(parents=True, exist_ok=True)
            pathlib.Path(config.ckpt_dir).mkdir(parents=True, exist_ok=True)
            pathlib.Path(config.code_dir).mkdir(parents=True, exist_ok=True)

            opt = self.opt
            config = self.config
        
            # dump the config to one file
            cfg_path = os.path.join(config.log_dir, "config.json")
            with open(cfg_path, 'w') as f:
                json.dump(vars(opt), f, indent=2)
                json.dump(config, f, indent=2)
                os.system('cp %s %s' % (opt.cfg, config.log_dir))
            config.cfg_path = cfg_path

            # set up logging
            self.setup_logger()
            logging.info(config)
            # init wandb *FIRST*
            if config.wandb.use_wandb:
                assert config.wandb.entity is not None
                Wandb.launch(config, self.opt, config.wandb.use_wandb)
                logging.info(f"Launch wandb, entity: {config.wandb.entity}")


    def setup_logger(self):
        """
        Configure logger on given level. Logging will occur on standard
        output and in a log file saved in model_dir.
        """
        loglevel = self.config.get('loglevel', 'INFO')  # Here, give a default value if there is no definition
        numeric_level = getattr(logging, loglevel.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: {}'.format(loglevel))

        log_format = logging.Formatter('%(asctime)s %(message)s')
        logger = logging.getLogger()
        logger.setLevel(numeric_level)

        file_handler = logging.FileHandler(osp.join(self.config.log_dir,
                                                    f'{osp.basename(self.config.log_dir)}.log'))
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

        file_handler = logging.StreamHandler(sys.stdout)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
        logging.root = logger
        logging.info(f"save log, checkpoint and code to: {self.config.log_dir}")
