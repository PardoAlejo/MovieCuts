import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
import torchvision
import sys
import os
sys.path.insert(1, f'{os.getcwd()}/models')
from video_resnet import r2plus1d_18
from audio_model import AVENet
from audio_visual_model import AudioVisualModel
from DB_loss import ResampleLoss
from callbacks import *
from cut_type_dataset import CutTypeDataset
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning as pl
import transforms as T
from scheduler import WarmupMultiStepLR
from torchvision.io import write_video
import json
from callbacks import MultilabelAP
import logging 

def sigmoid(X):
    return 1/(1+torch.exp(-X.squeeze()))

def get_transforms(config):
    
    transform_train = torchvision.transforms.Compose(        [
                T.ToTensorVideo(),
                T.RandomHorizontalFlipVideo(p=0.5),
                T.Resize((config.data.scale_h, config.data.scale_w)),
                T.NormalizeVideo(
                    mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989)
                ),
                T.RandomCropVideo((config.data.crop_size, config.data.crop_size)),
            ]
        )

    transform_val = torchvision.transforms.Compose(
        [
            T.ToTensorVideo(),
            T.RandomHorizontalFlipVideo(p=0.5),
            T.Resize((config.data.scale_h, config.data.scale_w)),
            T.NormalizeVideo(
                mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989)
            ),
            T.CenterCropVideo((config.data.crop_size, config.data.crop_size)),
        ]
    )
    return transform_train, transform_val

def get_dataloader(config):
    
    transforms_train, transforms_val = get_transforms(config)

    train_dataset = CutTypeDataset([config.data.shots_file_train, config.data.shots_file_val],
                    config.data.cut_type_file_name_train,
                    videos_path=config.data.videos_path,
                    cache_path=config.data.cache_path,
                    visual_stream=config.model.visual_stream,
                    audio_stream=config.model.audio_stream,
                    sampling=config.data.window_sampling,
                    snippet_size=config.data.snippet_size,
                    data_percent=config.data.data_percent,
                    distribution=config.data.distribution,
                    transform=transforms_train)
    logging.info(f'Num samples for train: {len(train_dataset)}')

    val_dataset = CutTypeDataset([config.data.shots_file_train, config.data.shots_file_val],
                    config.data.cut_type_file_name_val,
                    videos_path=config.data.videos_path,
                    cache_path=config.data.cache_path,
                    visual_stream=config.model.visual_stream,
                    audio_stream=config.model.audio_stream,
                    sampling=config.data.window_sampling,
                    snippet_size=config.data.snippet_size,
                    transform=transforms_val)

    logging.info(f'Num samples for val: {len(val_dataset)}')

    test_dataset = CutTypeDataset([config.data.shots_file_train, config.data.shots_file_val],
                    config.data.cut_type_file_name_test,
                    videos_path=config.data.videos_path,
                    cache_path=config.data.cache_path,
                    visual_stream=config.model.visual_stream,
                    audio_stream=config.model.audio_stream,
                    sampling=config.data.window_sampling,
                    snippet_size=config.data.snippet_size,
                    transform=transforms_val)
    
    train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=True)

    val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
            shuffle=False)

    test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
            shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


class Model(pl.LightningModule):
    
    def __init__(self, config, world_size):
        super().__init__()
        self.config = config

        self.beta_visual = config.model.vbeta
        self.beta_audio = config.model.abeta
        self.beta_audio_visual = config.model.avbeta

        self.batch_size = config.batch_size

        self._train_dataloader, self._val_dataloader, self._test_dataloader = get_dataloader(config)

        self.cut_types = self._train_dataloader.dataset.cut_types
        self.num_classes = len(self.cut_types)
        self.params = None
        self.initialization = config.training.initialization
        self.world_size = world_size
        self.lr = config.lr_scheduler.initial_lr * self.world_size

        self.visual_stream = config.model.visual_stream
        self.audio_stream = config.model.audio_stream
        self.video_model_path = config.training.video_model_path
        self.audio_model_path = config.training.audio_model_path

        if self.visual_stream and not self.audio_stream:

            self.r2p1d = r2plus1d_18(num_classes=self.num_classes)
            self.r2p1d.stem.requires_grad_(False)
            self.params = self.r2p1d.parameters()
            

        if not self.visual_stream and self.audio_stream:

            self.resnet18 = AVENet(model_depth=18, num_classes=self.num_classes)
            self.params = self.resnet18.parameters()

        if self.visual_stream and self.audio_stream:
            self.audio_visual_network = AudioVisualModel(num_classes=self.num_classes, mlp=True)
            self.params = self.audio_visual_network.parameters()
             
        if self.initialization == 'supervised':
            self._load_weights_supervised()
            logging.info(f'Loaded models from {config.training.video_model_path} and/or {config.training.audio_model_path}')
        else:
            logging.info('Training from scratch')


        self.ap_per_class_val = MultilabelAP(num_classes=self.num_classes, compute_on_step=False)
        self.ap_per_class_test = MultilabelAP(num_classes=self.num_classes, compute_on_step=False)

        self.save_hyperparameters()

        self.inference_logits_epoch = []
        self.clip_names_epoch = []

        if 'dbloss' in config.file:
            dbloss_args = config.dbloss
            focal = dict(focal=dbloss_args.focal_on, balance_param=dbloss_args.focal_balance, gamma=dbloss_args.focal_gamma)
            CB_loss = dict(CB_beta=dbloss_args.CB_beta, CB_mode=dbloss_args.CB_mode)
            map_param = dict(alpha=dbloss_args.map_alpha, beta=dbloss_args.map_beta, gamma=dbloss_args.map_gamma)
            logit_reg = dict(neg_scale=dbloss_args.logit_neg_scale, init_bias=dbloss_args.logit_init_bias)
            self.db_loss = ResampleLoss(use_sigmoid=True, 
                                    reduction='mean', 
                                    reweight_func='rebalance',
                                    focal=focal,
                                    CB_loss=CB_loss,
                                    map_param=map_param,
                                    logit_reg=logit_reg,
                                    device=self.device)
            self.model_loss = self.db_loss

        else:
            self.model_loss = self.bce_loss

    def forward(self, x):
        if self.visual_stream and not self.audio_stream:
            predictions = self.r2p1d(x)
        elif not self.visual_stream and self.audio_stream:
            predictions = self.resnet18(x)
        elif self.visual_stream and self.audio_stream:
            predictions = self.audio_visual_network(x)
        return predictions

    def _load_weights_supervised(self):
        
        if self.visual_stream and not self.audio_stream:
            state = torch.load(self.video_model_path)
            state_dict = self.r2p1d.state_dict()
            for k, v in state.items():
                if 'fc' in k:
                    continue
                state_dict.update({k: v})
            self.r2p1d.load_state_dict(state_dict)
            logging.info(f'Loaded visual weights from: {self.video_model_path}')

        elif self.audio_stream and not self.visual_stream:
            state = torch.load(self.audio_model_path)['model_state_dict']
            state_dict = self.resnet18.state_dict()
            
            for k, v in state.items():
                if 'fc' in k:
                    continue
                state_dict.update({k: v})
            self.resnet18.load_state_dict(state_dict)
        
            logging.info(f'Loaded audio weights from: {self.audio_model_path}')

        elif self.audio_stream and self.visual_stream:
            state_visual = torch.load(self.video_model_path)
            state_audio = torch.load(self.audio_model_path)['model_state_dict']
            state_dict = self.audio_visual_network.state_dict()
            
            for k, v in state_visual.items():
                if 'fc' in k:
                    continue
                state_dict.update({'r2p1d.'+k: v})
            
            for k, v in state_audio.items():
                if 'fc' in k:
                    continue
                state_dict.update({k: v})
            
            self.audio_visual_network.load_state_dict(state_dict)
    
    def training_step(self, batch, batch_idx):

        if self.visual_stream and not self.audio_stream:
            (video_chunk, labels, clip_names) = batch
            logits = self.r2p1d(video_chunk)
            loss = self.model_loss(logits, labels)
        elif not self.visual_stream and self.audio_stream:
            (audio_chunk, labels, clip_names) = batch
            logits = self.resnet18(audio_chunk)
            loss = self.model_loss(logits, labels)
        elif self.audio_stream and self.visual_stream:
            (video_chunk, audio_chunk, labels, clip_names) = batch
            logits, out_video, out_audio = self.audio_visual_network(video_chunk, audio_chunk)
            loss_audio = self.model_loss(out_audio, labels)
            loss_video = self.model_loss(out_video, labels)
            loss_multi = self.model_loss(logits, labels)

            loss = self.beta_audio_visual*loss_multi \
                    + self.beta_visual*loss_video \
                    + self.beta_audio*loss_audio

            self.log('Traning_loss_audio', loss_audio, 
                    on_step=True, 
                    on_epoch=True, 
                    logger=True)
            self.log('Traning_loss_video', loss_video, 
                    on_step=True, 
                    on_epoch=True, 
                    logger=True)
            self.log('Traning_loss_multi', loss_multi, 
                    on_step=True, 
                    on_epoch=True, 
                    logger=True)

        
        self.log('Traning_loss', loss, 
                    on_step=True, 
                    on_epoch=True, 
                    prog_bar=True, 
                    logger=True)
            
        return loss

    def validation_step(self, batch, batch_idx):

        if self.visual_stream and not self.audio_stream:
            (video_chunk, labels, clip_names) = batch
            logits = self.r2p1d(video_chunk)
            loss = self.model_loss(logits, labels)
        elif not self.visual_stream and self.audio_stream:
            (audio_chunk, labels, clip_names) = batch
            logits = self.resnet18(audio_chunk)
            loss = self.model_loss(logits, labels)

        elif self.audio_stream and self.visual_stream:
            (video_chunk, audio_chunk, labels, clip_names) = batch
            logits, out_video, out_audio = self.audio_visual_network(video_chunk, audio_chunk)
            loss_audio = self.model_loss(out_audio, labels)
            loss_video = self.model_loss(out_video, labels)
            loss_multi = self.model_loss(logits, labels)

            loss = self.beta_audio_visual*loss_multi \
                    + self.beta_visual*loss_video \
                    + self.beta_audio*loss_audio

            self.log('Validation_loss_audio', loss_audio, 
                    on_step=True, 
                    on_epoch=True, 
                    logger=True)
            self.log('Validation_loss_video', loss_video, 
                    on_step=True, 
                    on_epoch=True, 
                    logger=True)
            self.log('Validation_loss_multi', loss_multi, 
                    on_step=True, 
                    on_epoch=True, 
                    logger=True)

        self.log('Validation_loss', 
                loss,
                on_epoch=True,
                prog_bar=True, 
                logger=True)

        labels_metric = labels/(labels.max(dim=1)[0].unsqueeze(-1))

        self.ap_per_class_val.update(sigmoid(logits), labels_metric)
        
        self.log('Validation_mAP',
                 self.ap_per_class_val.compute()[0],
                 on_epoch=True,
                 prog_bar=True, 
                 logger=True)

        return logits, labels

    def test_step(self, batch, batch_idx):
        
        if self.visual_stream and not self.audio_stream:
            (video_chunk, labels, clip_names) = batch
            logits = self.r2p1d(video_chunk)
            loss = self.model_loss(logits, labels)
        elif not self.visual_stream and self.audio_stream:
            (audio_chunk, labels, clip_names) = batch
            logits = self.resnet18(audio_chunk)
            loss = self.model_loss(logits, labels)

        elif self.audio_stream and self.visual_stream:
            (video_chunk, audio_chunk, labels, clip_names) = batch
            logits, out_video, out_audio = self.audio_visual_network(video_chunk, audio_chunk)
            loss_audio = self.model_loss(out_audio, labels)
            loss_video = self.model_loss(out_video, labels)
            loss_multi = self.model_loss(logits, labels)

            loss = self.beta_audio_visual*loss_multi \
                    + self.beta_visual*loss_video \
                    + self.beta_audio*loss_audio

            self.log('Test_loss_audio', loss_audio, 
                    on_step=True, 
                    on_epoch=True, 
                    logger=True)
            self.log('Test_loss_video', loss_video, 
                    on_step=True, 
                    on_epoch=True, 
                    logger=True)
            self.log('Test_loss_multi', loss_multi, 
                    on_step=True, 
                    on_epoch=True, 
                    logger=True)

        self.log('Test_loss', 
                loss,
                on_epoch=True,
                prog_bar=True, 
                logger=True)

        labels_metric = labels/(labels.max(dim=1)[0].unsqueeze(-1))
        
        self.ap_per_class_test.update(sigmoid(logits), labels_metric)
        self.inference_logits_epoch.append(logits)
        self.clip_names_epoch.append(clip_names)

        return logits, labels

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.params, 
                                    lr=self.lr, 
                                    momentum=self.config.optimizer.momentum, 
                                    weight_decay=self.config.optimizer.weight_decay)

        warmup_iters = self.config.lr_scheduler.warmup_epochs * len(self._train_dataloader)
        lr_milestones = [len(self._train_dataloader) * m for m in self.config.lr_scheduler.lr_milestones]

        lr_scheduler ={'scheduler': WarmupMultiStepLR(optimizer,
                                    milestones=lr_milestones,
                                    gamma=self.config.lr_scheduler.lr_gamma,
                                    warmup_iters=warmup_iters,
                                    warmup_factor=0.5),
                    'name': 'lr',
                    "interval": "epoch",
                    "frequency": 1} 

        return [optimizer], [lr_scheduler]

    def bce_loss(self, logits, labels):
        bce = F.binary_cross_entropy_with_logits(
                logits,
                labels.type_as(logits))
        return bce

    def cross_entropy_loss(self, logits, labels):
        ce = F.cross_entropy(
            logits,
            labels.type_as(logits))
        return ce

    def focal_loss(self, logits, labels):
        p = sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, labels.type_as(logits), reduction="none")
        p_t = p * labels + (1 - p) * (1 - labels)
        loss = ce_loss * ((1 - p_t) ** self.config.focal_loss.gamma)
        return loss.mean()

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        if self.config.inference.test:
            return self._test_dataloader
        elif self.config.inference.validation:
            return self._val_dataloader
