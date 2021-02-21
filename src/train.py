import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
import torchvision
from torchvision.models.video import r2plus1d_18
from dataset import MovieDataset
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
import pytorch_lightning as pl
import transforms as T
from scheduler import WarmupMultiStepLR
from parameters import get_params

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

def generate_experiment_name(args):
    return f'experiment_sample-per-vid-{args.candidates_per_scene}'\
            f'_lr-{args.initial_lr}'\
            f'_val-neg-ratio-{args.negative_positive_ratio_val}'\
            f'_batchsize-{args.batch_size}'\
            f'_seed-{args.seed}'

def get_dataloader(args):
    
    transforms_train, transforms_val = get_transforms(args)

    train_dataset = MovieDataset(args.shots_file_name_train, 
                    transform=transforms_train,
                    num_positives_per_scene=args.candidates_per_scene,
                    size=(args.scale_w, args.scale_h))
    print(f'Num samples for train: {len(train_dataset)}')

    val_dataset = MovieDataset(args.shots_file_name_val,
                    transform=transforms_val,
                    num_positives_per_scene=args.candidates_per_scene, 
                    negative_positive_ratio=args.negative_positive_ratio_val,
                    size=(args.scale_w, args.scale_h))

    print(f'Num samples for val: {len(val_dataset)}')

    train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False)

    val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False)

    return train_dataloader, val_dataloader


class Model(pl.LightningModule):
    
    def __init__(self, args, world_size):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.r2p1d = r2plus1d_18(num_classes=self.args.num_classes)
        
        self._train_dataloader, self._val_dataloader = get_dataloader(args)

        self.accuracy = pl.metrics.Accuracy()
        
        self.world_size = world_size
        self.lr = self.args.initial_lr * self.world_size

        if not self.args.from_scratch:
            self._load_pretrained()
            self.params = self._set_layers_lr()
        else:
            self.params = self.r2p1d.parameters()

        self.save_hyperparameters()

    def forward(self, x):
        import ipdb; ipdb.set_trace()
        predictions = self.r2p1d(x)
        return predictions

    def _load_pretrained(self):
        state = torch.load(self.args.model_path)
        state_dict = self.r2p1d.state_dict()
        for k, v in state.items():
            if 'fc' in k:
                continue
            state_dict.update({k: v})
        self.r2p1d.load_state_dict(state_dict)

    def _set_layers_lr(self):

        params = []
        for name, child in self.r2p1d.named_children():
            if name == 'stem':
                this_params = {"params": child.parameters(), "lr": 0}
            elif name == 'fc':
                this_params = {"params": child.parameters(), "lr": self.args.fc_lr * args.world_size}
            else:
                this_params = {"params": child.parameters(), "lr": self.args.initial_lr * args.world_size}
            params.append(this_params)
        return params

    def training_step(self, batch, batch_idx):
        (video_chunk, labels) = batch
        logits = self.r2p1d(video_chunk)
        loss = self.bce_loss(logits, labels)
        self.log('Traning_loss', loss, 
                    on_step=True, 
                    on_epoch=True, 
                    prog_bar=True, 
                    logger=True)
        self.log('Training_Accuracy', 
                    self.accuracy(sigmoid(logits), labels),
                    prog_bar=True, 
                    on_epoch=True, 
                    on_step=False, 
                    logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (video_chunk, labels) = batch
        logits = self.r2p1d(video_chunk)
        loss = self.bce_loss(logits, labels)
        self.log('Validation_loss', 
                loss, 
                prog_bar=True, 
                logger=True)

        self.log('Validation_Accuracy', 
                self.accuracy(sigmoid(logits), labels), 
                prog_bar=True,
                logger=True)
    
    def test_step(self, batch, batch_idx):
        (video_chunk, labels) = batch
        logits = self.r2p1d(video_chunk)
        loss = self.bce_loss(logits, labels)
        self.log('Test_loss', 
                loss, 
                prog_bar=True, 
                logger=True)

        self.log('Test_Accuracy', 
                self.accuracy(sigmoid(logits), labels), 
                prog_bar=True,
                logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.params, 
                                    lr=self.lr, 
                                    momentum=self.args.momentum, 
                                    weight_decay=self.args.weight_decay)

        warmup_iters = self.args.lr_warmup_epochs * len(self._train_dataloader)
        lr_milestones = [len(self._train_dataloader) * m for m in self.args.lr_milestones]

        lr_scheduler = WarmupMultiStepLR(optimizer,
                                        milestones=lr_milestones,
                                        gamma=args.lr_gamma,
                                        warmup_iters=warmup_iters,
                                        warmup_factor=1e-5)
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

    early_stop_callback = EarlyStopping(
    monitor='Validation_loss',
    min_delta=0.00,
    patience=2,
    verbose=False,
    mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='step')

    experiment_name = generate_experiment_name(args)
    tb_logger = pl_loggers.TensorBoardLogger(args.experiments_dir, name=experiment_name)
    

    trainer = pl.Trainer(gpus=-1,
                        benchmark=True,
                        sync_batchnorm=True,
                        accelerator='ddp',
                        check_val_every_n_epoch=1,
                        progress_bar_refresh_rate=5,
                        weights_summary='top',
                        max_epochs=10,
                        logger=tb_logger,
                        callbacks=[lr_monitor],
                        profiler="simple",
                        num_sanity_val_steps=2) 

    model = Model(args, world_size=trainer.num_gpus)

    if args.test:
        path = args.checkpoint
        model_test = Model.load_from_checkpoint(path, args=args, world_size=trainer.num_gpus)
        print(f'Testing model from: {path}')
        trainer.test(model_test)
    else:
        trainer.fit(model)