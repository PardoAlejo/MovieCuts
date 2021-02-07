import numpy as np
import argparse
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
from torchvision.models.video import r2plus1d_18
from dataset import MovieDataset
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
import pytorch_lightning as pl

def sigmoid(X):
    return 1/(1+np.exp(-X))

def generate_experiment_name(args):
    return f'experiment_sample-per-vid-{args.candidates_per_sample}'\
            f'_lr-{args.initial_lr}'\
            f'_val-neg-ratio-{args.negative_positive_ratio_val}'\
            f'_batchsize-{args.batch_size}'\
            f'_seed-{args.seed}_layer-2-frozen'

def get_dataloader(args):
    train_dataset = MovieDataset(args.shots_file_name_train, 
                    num_positives_per_scene=args.candidates_per_sample)

    val_dataset = MovieDataset(args.shots_file_name_val, 
                    num_positives_per_scene=args.candidates_per_sample, 
                    negative_positive_ratio=args.negative_positive_ratio_val)

    train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False
            )
    val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False
            )

    return train_dataloader, val_dataloader


class Model(pl.LightningModule):
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.r2p1d = r2plus1d_18(num_classes=self.args.num_classes)
        
        self._train_dataloader, self._val_dataloader = get_dataloader(args)

        self.accuracy = pl.metrics.Accuracy()

        if not self.args.from_scratch:
            state = torch.load(self.args.model_path)
            state_dict = self.r2p1d.state_dict()
            for k, v in state.items():
                if 'fc' in k:
                    continue
                state_dict.update({k: v})
            self.r2p1d.load_state_dict(state_dict)

        for name, child in self.r2p1d.named_children():
            if name in ['layer3','layer4','avgpool','fc']:
                print(name + ' is unfrozen')
                for param in child.parameters():
                    param.requires_grad = True
            else:
                #stem layer1 'layer2' frozen
                print(name + ' is frozen')
                for param in child.parameters():
                    param.requires_grad = False


    def forward(self, x):
        predictions = self.r2p1d(x)
        return predictions
    
        
    def training_step(self, batch, batch_idx):
        (video_chunk, labels) = batch
        logits = self.r2p1d(video_chunk)
        loss = self.bce_loss(logits, labels)
        self.log('Traning_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('Training_Accuracy', self.accuracy(logits, labels), on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        (video_chunk, labels) = batch
        logits = self.r2p1d(video_chunk)
        loss = self.bce_loss(logits, labels)
        self.log('Validation_loss', loss, prog_bar=True, logger=True)
        self.log('Validation_Accuracy', self.accuracy(logits, labels), on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = Adam(self.r2p1d.parameters(), lr=self.args.initial_lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=self.args.lr_decay, patience=self.args.lr_patience, verbose=True)
        return {
       'optimizer': optimizer,
       'lr_scheduler': scheduler,
       'monitor': 'Validation_loss'
        }

    def bce_loss(self, logits, labels):
        bce = F.binary_cross_entropy_with_logits(logits.squeeze(),labels.type_as(logits))
        total_loss = bce
        return total_loss

    def train_dataloader(self):
        return self._train_dataloader


    def val_dataloader(self):
        return self._val_dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Easy video feature extractor')
    
    parser.add_argument('--shots_file_name_train', type=str, default='../data/used_cuts_train.csv',
                        help='Shots for training')
    parser.add_argument('--shots_file_name_val', type=str, default='../data/used_cuts_val.csv',
                        help='Shots for validation')
    parser.add_argument('--candidates_per_sample', type=int, default=10,
                        help='Number of candidates per sample')
    parser.add_argument('--negative_positive_ratio_val', type=int, default=5,
                        help='Ratio for negatives:positives for validation')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--initial_lr', type=float, default=0.0002,
                        help='Starting lr')
    parser.add_argument('--lr_decay', type=float, default=0.9,
                        help='Lr decay')
    parser.add_argument('--lr_patience', type=int, default=1,
                        help='iteration patience to reduce LR')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers for data loading')
    parser.add_argument('--from_scratch', action='store_false',
                        help='Start training from scratct,' 
                        ' starting from K-400 by default')
    parser.add_argument('--model_path', type=str, 
                        default='../models/r2plus1d_18-91a641e6.pth',
                        help='pretrained K400 model path')
    parser.add_argument('--num_classes', type=int, default=1,
                        help='Number of classes')
    parser.add_argument('--seed', type=int, default=4165,
                        help='Number of classes')
    parser.add_argument('--experiments_dir', type=str, default='../experiments',
                        help='Number of classes')
                        
    args = parser.parse_args()
    pl.utilities.seed.seed_everything(args.seed)

    early_stop_callback = EarlyStopping(
    monitor='Validation_loss',
    min_delta=0.00,
    patience=2,
    verbose=False,
    mode='min'
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    experiment_name = generate_experiment_name(args)
    tb_logger = pl_loggers.TensorBoardLogger(args.experiments_dir, name=experiment_name)
    

    trainer = pl.Trainer(gpus=-1,
                        accelerator='ddp',
                        check_val_every_n_epoch=1,
                        progress_bar_refresh_rate=5,
                        weights_summary='top',
                        max_epochs=100,
                        logger=tb_logger,
                        callbacks=[early_stop_callback, lr_monitor])
                        #,num_sanity_val_steps=0) remove santity check at some point

    model = Model(args)
    

    trainer.fit(model)

