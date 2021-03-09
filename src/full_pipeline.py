import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from callbacks import *
from pretrain import ModelPretrain, generate_experiment_name_pretrain
from finetune import ModelFinetune, generate_experiment_name_finetune
from parameters import get_params
from pytorch_lightning import loggers as pl_loggers
import os

if __name__ == "__main__":

    # Pretraining 
    args = get_params()
    print('Pretraining with args')
    print(args)

    experiment_name_pretrain = generate_experiment_name_pretrain(args)
    if args.initialization == 'pretrain':
        pretrain_experiment = f'{args.experiments_dir}/{experiment_name_pretrain}/'
    else:
        pretrain_experiment = f'{args.experiments_dir}/{args.initialization}_audio_{args.audio_stream}_visual_{args.visual_stream}'

    print(f'Logs path: {pretrain_experiment}')

    if not os.path.exists(f'{pretrain_experiment}/last.ckpt') and args.initialization == 'pretrain':
        pl.utilities.seed.seed_everything(args.seed)

        lr_monitor_pretrain = LearningRateMonitor(logging_interval='step')

        
        tb_logger_pretrain = pl_loggers.TensorBoardLogger(args.experiments_dir, name=experiment_name_pretrain, version=0)
        
        ModelCheckpointPretrain = ModelCheckpoint(
                                    dirpath=pretrain_experiment,
                                    filename='epoch-{epoch}-_ValAcc-{Validation_Accuracy:1.2f}',
                                    save_last=True,
                                    period=1,
                                    )

        pretrainer = pl.Trainer(gpus=-1,
                            accelerator='ddp',
                            check_val_every_n_epoch=1,
                            progress_bar_refresh_rate=1,
                            weights_summary='top',
                            max_epochs=args.pretrain_max_epochs,
                            logger=tb_logger_pretrain,
                            callbacks=[lr_monitor_pretrain, ModelCheckpointPretrain],
                            profiler="simple",
                            num_sanity_val_steps=0) 

        print(f"Using {pretrainer.num_gpus} gpus")
        model_pretrain = ModelPretrain(args, world_size=pretrainer.num_gpus)
        print(f'Pretraining model with audio: {args.audio_stream} and visual: {args.visual_stream}')
        pretrainer.fit(model_pretrain)
    else:
        print('Pretrain model found, finetuning now')
        pass

    # Finetuning
    print('Finetuning')

    if args.epoch:
        args.pretrain_model_path = f'{pretrain_experiment}/version_0/checkpoints/epoch-{args.finetune_epoch}.ckpt'
    else:
        args.pretrain_model_path = f'{pretrain_experiment}/last.ckpt'
    pl.utilities.seed.seed_everything(args.seed)

    lr_monitor_finetuning = LearningRateMonitor(logging_interval='step')

    experiment_name_finetune = generate_experiment_name_finetune(args)
    tb_logger_finetune = pl_loggers.TensorBoardLogger(pretrain_experiment, name=experiment_name_finetune)

    ModelCheckpointFinetune = ModelCheckpoint(
                                    dirpath=f'{pretrain_experiment}/{experiment_name_finetune}',
                                    monitor='Validation_mAP',
                                    filename='epoch-{epoch}_ValmAP-{Validation_mAP:1.2f}',
                                    save_top_k=2,
                                    mode='max',
                                    period=2
                                    )

    callbacks=[lr_monitor_finetuning, ModelCheckpointFinetune, WriteMetricReport()]
    trainer_finetune = pl.Trainer(gpus=-1,
                        accelerator='ddp',
                        check_val_every_n_epoch=1,
                        progress_bar_refresh_rate=1,
                        weights_summary='top',
                        max_epochs=args.finetune_max_epochs,
                        logger=tb_logger_finetune,
                        callbacks=callbacks,
                        profiler="simple",
                        num_sanity_val_steps=0) 

    print(f"Using {trainer_finetune.num_gpus} gpus")
    model_finetune = ModelFinetune(args, world_size=trainer_finetune.num_gpus)


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
        print(f'Finetuning model with audio: {args.audio_stream} and visual: {args.visual_stream}')
        trainer_finetune.fit(model_finetune)


