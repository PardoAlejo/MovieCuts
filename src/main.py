from training import *
import argparse
import logging, os.path as osp
sys.path.insert(1, f'{os.getcwd()}/utils')
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

def parse_option():
    parser = argparse.ArgumentParser('MovieCuts for cut recognition training')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    args, opts = parser.parse_known_args()
    config.load(args.cfg, recursive=True)
    config.update(opts)
    config.file = args.cfg
    return args, config    

def generate_exp_directory(config):
    config.exp_dir = f'{config.base_exp_dir}/'\
           f'{config.training.initialization}'\
           f'_audio_{config.model.audio_stream}'\
           f'_visual_{config.model.visual_stream}'

def generate_exp_name(config, tags, name=None, logname=None):
    """Function to create checkpoint folder.

    Args:
        config:
        tags: tags for saving and generating the expname
        name: specific name for experiment (optional)
        logname: the name for the current run. None if auto
    Returns:
        the expname
    """

    if logname is None:
        logname = ''.join(tags[:-2])
        if name:
            logname = '-'.join([name])
    config.exp_name = logname

def main(opt, config):    
    # ------- PL Trainer ----------
    if config.mode.train:
        version_num = get_experiment_version(config)
        logger = pl_loggers.TensorBoardLogger(save_dir=config.exp_dir, name=config.exp_name, version=version_num)

        pl.utilities.seed.seed_everything(config.seed)

        # ---------- Callbacks --------------
        wandb_config_cb = wandb_config(opt, config)
        # -- LR Monitor --
        lr_monitor = LearningRateMonitor(logging_interval='step')

        # -- Checkpointing --
        ModelCheckpointCB = ModelCheckpoint(
                                        monitor='Validation_loss',
                                        dirpath=osp.join(config.exp_dir, config.exp_name, f'version_{version_num}', 'checkpoints'),
                                        filename='{epoch}_{Validation_loss:1.2f}',
                                        save_top_k=config.training.save_top_k,
                                        mode='min',
                                        every_n_epochs=config.training.save_freq
                                        )
        
        callbacks_train=[wandb_config_cb, lr_monitor, ModelCheckpointCB, WriteMetricReport()]
        trainer = pl.Trainer(gpus=-1,
                            accelerator='ddp',
                            check_val_every_n_epoch=config.training.val_freq,
                            progress_bar_refresh_rate=config.training.print_freq,
                            weights_summary='top',
                            max_epochs=config.training.max_epochs,
                            logger=logger,
                            callbacks=callbacks_train,
                            profiler="simple",
                            num_sanity_val_steps=config.training.sanity_check_steps) 
        logging.info(f"Using {trainer.num_gpus} gpus")

        model = Model(config, world_size=trainer.num_gpus)
        logging.info(f'Finetuning model with audio: {config.model.audio_stream} and visual: {config.model.visual_stream}')
        trainer.fit(model)

    elif config.mode.inference:
        
        if not config.inference.save_path:
            version_num = get_experiment_version(config)
            path_comps = os.path.normpath(config.inference.checkpoint).split(os.sep)
            exp_dir = '/'.join(path_comps[:-4])
            logging.info(path_comps)
            exp_name = path_comps[-4]
            version_num = path_comps[-3]
            logging.info(f'Running inference on model {exp_name} version {version_num} epoch {path_comps[-1]}')
        else:
            exp_dir=config.inference.save_path
            if not os.path.exists(exp_dir):
                os.makedirs(exp_dir)
            exp_name=os.path.splitext(os.path.basename(config.inference.checkpoint))[0]
            version_num=0
            logging.info(f'Running inference on model {config.inference.checkpoint}, saving outputs in {exp_dir}')

        logger = pl_loggers.TensorBoardLogger(save_dir=exp_dir, name=exp_name, version=version_num)
        tester = pl.Trainer(gpus=-1,
                        accelerator='ddp',
                        progress_bar_refresh_rate=config.inference.print_freq,
                        logger=logger,
                        callbacks=[WriteMetricReport(), SaveLogits()],
                        num_sanity_val_steps=config.inference.sanity_check_steps)

        path = config.inference.checkpoint
        model_test = Model.load_from_checkpoint(path, config=config, world_size=tester.num_gpus)
        data_partition = 'Test' if config.inference.test else 'Validation'
        logging.info(f'Forwarding on {data_partition} using model from: {path}')
        tester.test(model_test)
            

if __name__ == "__main__":
    opt, config = parse_option()
    print(config)
    # Generate experiment names and directories.
    tags = [f'{osp.splitext(osp.basename(opt.cfg))[0]}_',
            f'_snipsize-{config.data.snippet_size}',
            f'_cropsize-{config.data.crop_size}',
            f'_winsamp-{config.data.window_sampling}',
            f'_lr-{config.lr_scheduler.initial_lr}',
            f'_abeta-{config.model.abeta}',
            f'_vbeta-{config.model.vbeta}',
            f'_avbeta-{config.model.avbeta}',
            f'_bs-{config.batch_size}',
            f'_negatives-{config.data.negative_portion}',
            f'validation_set-{config.inference.validation}',
            f'test_set-{config.inference.test}']
    # -- Logger and Directories--
    if config.mode.train:
        generate_exp_directory(config)
        generate_exp_name(config, tags)
    
        config.wandb.tags = tags
        config.wandb.name = config.exp_name
    # run a toy example
    main(opt, config)

    