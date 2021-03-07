import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
import torchvision
import sys
sys.path.insert(1, '/home/pardogl/LTC-e2e/models')
print(sys.path)
from video_resnet import r2plus1d_18
from audio_model import AVENet
from audio_visual_model import AudioVisualModel
from cut_type_dataset import CutTypeDataset
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
import pytorch_lightning as pl
import transforms as T
from scheduler import WarmupMultiStepLR
from parameters import get_params
from torchvision.io import write_video
import json
from pytorch_lightning.loggers.csv_logs import CSVLogger 

def sigmoid(X):
    return 1/(1+torch.exp(-X.squeeze()))

def get_transforms(args):
    
    transform_train = torchvision.transforms.Compose(        [
                T.ToTensorVideo(),
                T.Resize((args.scale_h, args.scale_w)),
                T.NormalizeVideo(
                    mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989)
                ),
                T.RandomCropVideo((args.crop_size, args.crop_size)),
            ]
        )

    transform_val = torchvision.transforms.Compose(
        [
            T.ToTensorVideo(),
            T.Resize((args.scale_h, args.scale_w)),
            T.NormalizeVideo(
                mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989)
            ),
            T.CenterCropVideo((args.crop_size, args.crop_size)),
        ]
    )
    return transform_train, transform_val

def generate_experiment_name_finetune(args):
    return f'cut-type_'\
            f'data-percent_{args.finetune_data_percent}'\
            f'_initialization-{args.initialization}'\
            f'_epoch-{args.epoch}' \
            f'_lr-{args.finetune_initial_lr}'\
            f'_batchsize-{args.finetune_batch_size}'

def get_dataloader(args):
    
    transforms_train, transforms_val = get_transforms(args)

    train_dataset = CutTypeDataset(args.shots_file_names,
                    args.cut_type_file_name_train,
                    visual_stream=args.visual_stream,
                    audio_stream=args.audio_stream,
                    snippet_size=args.snippet_size,
                    data_percent=args.finetune_data_percent,
                    transform=transforms_train)
    print(f'Num samples for train: {len(train_dataset)}')

    val_dataset = CutTypeDataset(args.shots_file_names,
                    args.cut_type_file_name_val,
                    visual_stream=args.visual_stream,
                    audio_stream=args.audio_stream,
                    snippet_size=args.snippet_size,
                    transform=transforms_val)

    print(f'Num samples for val: {len(val_dataset)}')

    test_dataset = CutTypeDataset(args.shots_file_names,
                    args.cut_type_file_name_test,
                    visual_stream=args.visual_stream,
                    audio_stream=args.audio_stream,
                    snippet_size=args.snippet_size,
                    transform=transforms_val)
    
    train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.finetune_batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False)

    val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.finetune_batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False)

    test_dataloader = DataLoader(
            val_dataset,
            batch_size=args.finetune_batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


class ModelFinetune(pl.LightningModule):
    
    def __init__(self, args, world_size):
        super().__init__()
        self.args = args
        self.batch_size = self.args.finetune_batch_size

        self._train_dataloader, self._val_dataloader, self._test_dataloader = get_dataloader(args)

        self.cut_types = self._train_dataloader.dataset.cut_types
        self.num_classes = len(self.cut_types)
        self.params = None
        self.initialization = self.args.initialization
        self.world_size = world_size
        self.lr = self.args.finetune_initial_lr * self.world_size

        if self.args.visual_stream and not self.args.audio_stream:

            self.r2p1d = r2plus1d_18(num_classes=self.num_classes)
            self.r2p1d.stem.requires_grad_(False)
            self.params = self.r2p1d.parameters()
            

        if not self.args.visual_stream and self.args.audio_stream:

            self.resnet18 = AVENet(model_depth=18, num_classes=self.num_classes)
            self.params = self.resnet18.parameters()

        if self.args.visual_stream and self.args.audio_stream:
            self.audio_visual_network = AudioVisualModel(num_classes=self.num_classes, mlp=True)
            self.params = self.audio_visual_network.parameters()
             
        if self.initialization == 'supervised':
            self._load_weights_supervised()
            print(f'Loaded models from {self.args.video_model_path} and/or {self.args.audio_model_path}')
        elif self.initialization == 'pretrain':
            self._load_weights_pretrain()
            print(f'Loaded models from {self.args.pretrain_model_path}')
        else:
            print('Training from scratch')
            
        self.accuracy = pl.metrics.Accuracy()
        self.ap = pl.metrics.AveragePrecision()
        self.save_hyperparameters()

    def forward(self, x):
        if self.args.visual_stream and not self.args.audio_stream:
            predictions = self.r2p1d(x)
        elif not args.visual_stream and self.args.audio_stream:
            self.resnet18(x)
        elif args.visual_stream and self.args.audio_stream:
            pass
        return predictions

    def _load_weights_supervised(self):
        
        if self.args.visual_stream and not self.args.audio_stream:
            state = torch.load(self.args.video_model_path)
            state_dict = self.r2p1d.state_dict()
            for k, v in state.items():
                if 'fc' in k:
                    continue
                state_dict.update({k: v})
            self.r2p1d.load_state_dict(state_dict)
            print(f'Loaded visual weights from: {self.args.video_model_path}')

        elif self.args.audio_stream and not self.args.visual_stream:
            state = torch.load(self.args.audio_model_path)['model_state_dict']
            state_dict = self.resnet18.state_dict()
            
            for k, v in state.items():
                if 'fc' in k:
                    continue
                state_dict.update({k: v})
            self.resnet18.load_state_dict(state_dict)
        
            print(f'Loaded audio weights from: {self.args.audio_model_path}')

        elif self.args.audio_stream and self.args.visual_stream:
            state_visual = torch.load(self.args.video_model_path)
            state_audio = torch.load(self.args.audio_model_path)['model_state_dict']
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
    
    def _load_weights_pretrain(self):
        
        state = torch.load(self.args.pretrain_model_path)['state_dict']

        if self.args.visual_stream and not self.args.audio_stream:
            state_dict = self.r2p1d.state_dict()
            for k, v in state.items():
                if 'fc' in k:
                    continue
                state_dict.update({k.replace('r2p1d.',''): v})
            self.r2p1d.load_state_dict(state_dict)
            print(f'Loaded visual weights from: {self.args.video_model_path}')

        elif self.args.audio_stream and not self.args.visual_stream:
            state_dict = self.resnet18.state_dict()
            
            for k, v in state.items():
                if 'fc' in k:
                    continue
                state_dict.update({k.replace('resnet18.',''): v})
            self.resnet18.load_state_dict(state_dict)
        
            print(f'Loaded audio weights from: {self.args.audio_model_path}')

        elif self.args.audio_stream and self.args.visual_stream:
            state_dict = self.audio_visual_network.state_dict()
            
            for k, v in state.items():
                if 'fc_aux' in k or 'fc_final' in k:
                    continue
                state_dict.update({k.replace('audio_visual_network.',''): v})

            self.audio_visual_network.load_state_dict(state_dict)

    def training_step(self, batch, batch_idx):

        if self.args.visual_stream and not self.args.audio_stream:
            (video_chunk, labels, clip_names) = batch
            logits = self.r2p1d(video_chunk)
            loss = self.bce_loss(logits, labels)
        elif not self.args.visual_stream and self.args.audio_stream:
            (audio_chunk, labels, clip_names) = batch
            logits = self.resnet18(audio_chunk)
            loss = self.bce_loss(logits, labels)
        elif self.args.audio_stream and self.args.visual_stream:
            (video_chunk, audio_chunk, labels, clip_names) = batch
            logits, out_video, out_audio = self.audio_visual_network(video_chunk, audio_chunk)
            loss_audio = self.bce_loss(out_audio, labels)
            loss_video = self.bce_loss(out_video, labels)
            loss_multi = self.bce_loss(logits, labels)

            loss = loss_multi + loss_video + loss_audio
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
        self.log('Training_AP', 
                    self.ap(sigmoid(logits), labels),
                    prog_bar=True, 
                    on_epoch=True, 
                    on_step=True, 
                    logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):

        if self.args.visual_stream and not self.args.audio_stream:
            (video_chunk, labels, clip_names) = batch
            logits = self.r2p1d(video_chunk)
            loss = self.bce_loss(logits, labels)
        elif not args.visual_stream and self.args.audio_stream:
            (audio_chunk, labels, clip_names) = batch
            logits = self.resnet18(audio_chunk)
            loss = self.bce_loss(logits, labels)

        elif self.args.audio_stream and self.args.visual_stream:
            (video_chunk, audio_chunk, labels, clip_names) = batch
            logits, out_video, out_audio = self.audio_visual_network(video_chunk, audio_chunk)
            loss_audio = self.bce_loss(out_audio, labels)
            loss_video = self.bce_loss(out_video, labels)
            loss_multi = self.bce_loss(logits, labels)

            loss = loss_multi + loss_video + loss_audio
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
                prog_bar=True, 
                logger=True)

        self.log('Validation_AP', 
                self.ap(sigmoid(logits), labels), 
                prog_bar=True,
                logger=True)

        return self.log

    def test_step(self, batch, batch_idx):

        if self.args.visual_stream and not self.args.audio_stream:
            (video_chunk, labels, clip_names) = batch
            logits = self.r2p1d(video_chunk)
            loss = self.bce_loss(logits, labels)
        elif not args.visual_stream and self.args.audio_stream:
            (audio_chunk, labels, clip_names) = batch
            logits = self.resnet18(audio_chunk)
            loss = self.bce_loss(logits, labels)

        elif self.args.audio_stream and self.args.visual_stream:
            (video_chunk, audio_chunk, labels, clip_names) = batch
            logits, out_video, out_audio = self.audio_visual_network(video_chunk, audio_chunk)
            loss_audio = self.bce_loss(out_audio, labels)
            loss_video = self.bce_loss(out_video, labels)
            loss_multi = self.bce_loss(logits, labels)

            loss = loss_multi + loss_video + loss_audio
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
                prog_bar=True, 
                logger=True)

        self.log('Test_AP', 
                self.ap(sigmoid(logits), labels), 
                prog_bar=True,
                logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.params, 
                                    lr=self.lr, 
                                    momentum=self.args.momentum, 
                                    weight_decay=self.args.weight_decay)

        warmup_iters = self.args.lr_warmup_epochs * len(self._train_dataloader)
        lr_milestones = [len(self._train_dataloader) * m for m in self.args.finetune_lr_milestones]

        lr_scheduler ={'scheduler': WarmupMultiStepLR(optimizer,
                                    milestones=lr_milestones,
                                    gamma=self.args.lr_gamma,
                                    warmup_iters=warmup_iters,
                                    warmup_factor=0.5),
                    'name': 'lr'} 

        return [optimizer], [lr_scheduler]

    def bce_loss(self, logits, labels):
        bce = F.binary_cross_entropy_with_logits(
                logits.squeeze(),
                labels.type_as(logits))
        return bce

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader


if __name__ == "__main__":

    args = get_params()
    print(args)
    pl.utilities.seed.seed_everything(args.seed)

    early_stop_callback = EarlyStopping(
    monitor='Validation_loss',
    min_delta=0.00,
    patience=2,
    verbose=False,
    mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='step')

    experiment_name = generate_experiment_name_finetune(args)
    tb_logger = pl_loggers.TensorBoardLogger(args.experiments_dir, name=experiment_name)
    csv_logger = CSVLogger(args.experiments_dir, name="test")

    trainer = pl.Trainer(gpus=-1,
                        accelerator='ddp',
                        check_val_every_n_epoch=1,
                        progress_bar_refresh_rate=1,
                        weights_summary='top',
                        max_epochs=args.max_epochs,
                        logger=csv_logger,
                        callbacks=[lr_monitor],
                        profiler="simple",
                        num_sanity_val_steps=0) 

    print(f"Using {trainer.num_gpus} gpus")
    model = ModelFinetune(args, world_size=trainer.num_gpus)

    if args.finetune_test:
        tester = pl.Trainer(gpus=-1,
                        accelerator='ddp',
                        progress_bar_refresh_rate=1,
                        weights_summary='top',
                        profiler="simple",
                        num_sanity_val_steps=0)

        path = args.finetune_checkpoint
        model_test = Model.load_from_checkpoint(path, args=args, world_size=tester.num_gpus)
        print(f'Testing model from: {path}')
        tester.test(model_test)
    else:
        print(f'Training model with audio: {args.audio_stream} and visual: {args.visual_stream}')
        trainer.fit(model)