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
from pretext_dataset import PretextDataset
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

def generate_experiment_name_pretrain(args):
    return f'pretrain_'\
            f'_from_scratch_{args.pretrain_from_scratch}'\
            f'_audio_{args.audio_stream}'\
            f'_visual_{args.visual_stream}'\
            f'_snippet_{args.snippet_size}'\
            f'_lr-{args.pretrain_initial_lr}'\
            f'_batchsize-{args.pretrain_batch_size}'\
            f'_seed-{args.seed}'

def get_dataloader(args):
    
    transforms_train, transforms_val = get_transforms(args)

    train_dataset = PretextDataset(args.shots_file_names[0],
                    visual_stream=args.visual_stream,
                    audio_stream=args.audio_stream,
                    snippet_size=args.snippet_size,
                    transform=transforms_train,
                    size=(args.scale_w, args.scale_h))
    print(f'Num samples for train: {len(train_dataset)}')

    val_dataset = PretextDataset(args.shots_file_names[1],
                    visual_stream=args.visual_stream,
                    audio_stream=args.audio_stream,
                    snippet_size=args.snippet_size,
                    transform=transforms_val,
                    negative_positive_ratio=args.negative_positive_ratio_val,
                    size=(args.scale_w, args.scale_h))

    print(f'Num samples for val: {len(val_dataset)}')

    train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.pretrain_batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False)

    val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.pretrain_batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False)

    return train_dataloader, val_dataloader


class ModelPretrain(pl.LightningModule):
    
    def __init__(self, args, world_size):
        super().__init__()
        self.args = args
        self.batch_size = self.args.pretrain_batch_size

        self.params = None
        
        self.world_size = world_size
        self.lr = self.args.pretrain_initial_lr * self.world_size

        if self.args.visual_stream and not self.args.audio_stream:

            self.r2p1d = r2plus1d_18(num_classes=self.args.pretrain_num_classes)
            self.r2p1d.stem.requires_grad_(False)
            if not self.args.pretrain_from_scratch:
                self._load_pretrained(stream='visual')
            self._set_layers_params_video()
            

        if not self.args.visual_stream and self.args.audio_stream:

            self.resnet18 = AVENet(model_depth=18, num_classes=self.args.pretrain_num_classes)
            if not self.args.pretrain_from_scratch:
                self._load_pretrained()
            self._set_layers_params_audio()
                   
        
        if self.args.visual_stream and self.args.audio_stream:
            self.audio_visual_network = AudioVisualModel(num_classes=self.args.pretrain_num_classes, mlp=True)
            if not self.args.pretrain_from_scratch:
                self._load_pretrained()
            self._set_layers_params_multi()
             

        self._train_dataloader, self._val_dataloader = get_dataloader(args)

        self.accuracy = pl.metrics.Accuracy()

        self.save_hyperparameters()

    def forward(self, x):
        if self.args.visual_stream and not self.args.audio_stream:
            predictions = self.r2p1d(x)
        elif not args.visual_stream and self.args.audio_stream:
            self.resnet18(x)
        elif args.visual_stream and self.args.audio_stream:
            pass
        return predictions

    def _load_pretrained(self, stream='visual'):
        
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

    def _set_layers_params_audio(self):
        self.params = self.resnet18.parameters()

    def _set_layers_params_multi(self):
        self.params = self.audio_visual_network.parameters()

    def _set_layers_params_video(self):
        self.params = [
            {"params": self.r2p1d.stem.parameters(), "lr": 0},
            {"params": self.r2p1d.layer1.parameters()},
            {"params": self.r2p1d.layer2.parameters()},
            {"params": self.r2p1d.layer3.parameters()},
            {"params": self.r2p1d.layer4.parameters()},
            {"params": self.r2p1d.fc.parameters(), "lr": self.lr},
        ]

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
        self.log('Training_Accuracy', 
                    self.accuracy(sigmoid(logits), labels),
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

        self.log('Validation_Accuracy', 
                self.accuracy(sigmoid(logits), labels), 
                prog_bar=True,
                logger=True)
    
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
            
            # dict_scores = {}
            # for idx, clip_name in enumerate(clip_names):
            #     if labels[idx] == 1:
            #         dict_scores[clip_name] = {'audio': out_audio[idx].cpu().numpy().item(), 'visual': out_video[idx].cpu().numpy().item(), 'audio-visual':logits[idx].cpu().numpy().item()}
            
            # with open(f'scores/val_scores_batch_{batch_idx}.json', 'w') as f:
            #     json.dump(dict_scores, f)

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

        self.log('Validation_Accuracy', 
                self.accuracy(sigmoid(logits), labels), 
                prog_bar=True,
                logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.params, 
                                    lr=self.lr, 
                                    momentum=self.args.momentum, 
                                    weight_decay=self.args.weight_decay)

        warmup_iters = self.args.lr_warmup_epochs * len(self._train_dataloader)
        lr_milestones = [len(self._train_dataloader) * m for m in self.args.pretrain_lr_milestones]

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
        return self._val_dataloader


if __name__ == "__main__":

    args = get_params()
    print(args)
    pl.utilities.seed.seed_everything(args.seed)

    lr_monitor = LearningRateMonitor(logging_interval='step')

    experiment_name = generate_experiment_name_pretrain(args)
    tb_logger = pl_loggers.TensorBoardLogger(args.experiments_dir, name=experiment_name)
    

    trainer = pl.Trainer(gpus=-1,
                        accelerator='ddp',
                        check_val_every_n_epoch=1,
                        progress_bar_refresh_rate=1,
                        weights_summary='top',
                        max_epochs=args.pretrain_max_epochs,
                        logger=tb_logger,
                        callbacks=[lr_monitor],
                        profiler="simple",
                        num_sanity_val_steps=0) 

    print(f"Using {trainer.num_gpus} gpus")
    model = ModelPretrain(args, world_size=trainer.num_gpus)

    if args.pretrain_test:
        tester = pl.Trainer(gpus=-1,
                        accelerator='ddp',
                        progress_bar_refresh_rate=1,
                        weights_summary='top',
                        profiler="simple",
                        num_sanity_val_steps=0)

        path = args.pretrain_checkpoint
        model_test = Model.load_from_checkpoint(path, args=args, world_size=tester.num_gpus)
        print(f'Testing model from: {path}')
        tester.test(model_test)
    else:
        print(f'Training model with audio: {args.audio_stream} and visual: {args.visual_stream}')
        trainer.fit(model)