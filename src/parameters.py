import argparse


def get_params():
    parser = argparse.ArgumentParser(description='Easy video feature extractor')

    parser.add_argument('--visual_stream', action='store_true',
                        help='Turn on video stream')
    parser.add_argument('--audio_stream', action='store_true',
                        help='Turn on video stream')

    parser.add_argument('--shots_file_names', nargs="+", 
                        default=['data/annotated_clips_train.csv', 'data/annotated_clips_val.csv'],
                        help='Shots info')
    
    # Reading arguments
    parser.add_argument("--scale_h", default=128, type=int,
                        help="Scale H to read")
    parser.add_argument("--scale_w", default=180, type=int,
                        help="Scale H to read")
    parser.add_argument("--crop_size", default=112, type=int, 
                        help="number of frames per clip")
    parser.add_argument("--snippet_size", default=16, type=int, 
                        help="number of frames per clip")

    parser.add_argument('--negative_positive_ratio_val', type=int, default=1,
                        help='Ratio for negatives:positives for validation')

    
    # Batch Size and initial learning rates
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers for data loading')
    parser.add_argument('--pretrain_batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument("--pretrain_initial_lr", default=0.001, type=float, 
                        help="initial learning rate")

    # Scheduler parameters
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="momentum")
    parser.add_argument('--lr_decay', type=float, default=0.9,
                        help='Lr decay')
    parser.add_argument("--weight-decay", default=1e-4, type=float, 
                        help="weight decay (default: 1e-4)")
    parser.add_argument("--pretrain_max_epochs", default=5, type=int,
                        help="Max number of epochs for training")
    parser.add_argument("--pretrain_lr-milestones",nargs="+",default=[3, 4],
                        type=int,help="decrease lr on milestones")
    parser.add_argument("--lr-gamma",default=0.5,type=float,
                        help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-warmup-epochs", default=1, type=int,
                        help="number of warmup epochs")

    parser.add_argument('--pretrain_from_scratch', action='store_true',
                        help='Start training from scratct,' 
                        ' starting from K-400 by default')
    parser.add_argument('--video_model_path', type=str, 
                        default='model_checkpoints/r2plus1d_18-91a641e6.pth',
                        help='pretrained K400 model path')
    parser.add_argument('--audio_model_path', type=str, 
                        default='model_checkpoints/vggsound_avgpool.pth.tar',
                        help='pretrained K400 model path')
    parser.add_argument('--pretrain_num_classes', type=int, default=1,
                        help='Number of classes')
    parser.add_argument('--seed', type=int, default=4165,
                        help='Number of classes')
    parser.add_argument('--experiments_dir', type=str, default='experiments',
                        help='Number of classes')
    #Loss Parameters

    parser.add_argument('--pretrain_vbeta', type=float, default=1,
                        help='Loss weight for visual pretrain')
    parser.add_argument('--pretrain_abeta', type=float, default=1,
                        help='Loss weight for audio pretrain')
    parser.add_argument('--pretrain_avbeta', type=float, default=1,
                        help='Loss weight for audio-visual pretrain')
    
    parser.add_argument('--finetune_vbeta', type=float, default=1,
                        help='Loss weight for visual finetuning')
    parser.add_argument('--finetune_abeta', type=float, default=1,
                        help='Loss weight for audio finetuning')
    parser.add_argument('--finetune_avbeta', type=float, default=1,
                        help='Loss weight for audio-visual finetuning')
    parser.add_argument('--gamma', type=float, default=0,
                        help='Gamma for focal loss')

    # Test or load checkpoint
    parser.add_argument('--pretrain_checkpoint', type=str, default='store_true',
                        help='Checkpoint to test or resume')
    parser.add_argument('--pretrain_test', action='store_true',
                        help='Checkpoint to test or resume')                    

    # Fine tune train set params
    parser.add_argument('--finetune_data_percent', type=float, default=1,
                        help='Percentage of data to finetune with')
    parser.add_argument('--distribution', type=str, default='natural',
                        choices=['natural', 'uniform', 'sqrt'],
                        help='Data distribution for training')

    parser.add_argument('--cut_type_file_name_train', type=str, default='data/cut-type-train.json',
                        help='Cut types for training')
    parser.add_argument('--cut_type_file_name_val', type=str, default='data/cut-type-val.json',
                        help='Cut types for validation')
    parser.add_argument('--cut_type_file_name_test', type=str, default='data/cut-type-test.json',
                        help='Cut types for testing')
                        
    # Batch Size and initial learning rates
    parser.add_argument('--finetune_batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument("--finetune_initial_lr", default=0.001, type=float, 
                        help="initial learning rate")

    # Scheduler parameters
    parser.add_argument("--finetune_max_epochs", default=8, type=int,
                        help="Max number of epochs for training")
    parser.add_argument("--finetune_lr-milestones",nargs="+",default=[4, 6],
                        type=int,help="decrease lr on milestones")
    parser.add_argument("--finetune_lr-gamma",default=0.5,type=float,
                        help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--finetune_lr-warmup-epochs", default=1, type=int,
                        help="number of warmup epochs")

    #Initialization Parameters
    parser.add_argument('--initialization', type=str, default='pretrain',
                        choices=['pretrain', 'scratch', 'supervised'],
                        help='Start training from scratch,' 
                        ' from pretrain task, or from Kinetics/VGGSound')
    parser.add_argument('--pretrain_model_path', type=str, 
                        default=None,
                        help='pretrained model path') 
    parser.add_argument('--epoch', type=str,
                        default=None,
                        help='Epoch to load weights from finetune')  

    # Test or load checkpoint
    parser.add_argument('--finetune_checkpoint', type=str, default='store_true',
                        help='Checkpoint to test or resume')
    parser.add_argument('--finetune_validation', action='store_true',
                        help='Checkpoint to validate or resume') 
    parser.add_argument('--finetune_test', action='store_true',
                        help='Checkpoint to test or resume')                    
    
    #DB Loss
    parser.add_argument('--focal_on', action='store_true')
    parser.add_argument('--focal_balance', type=float, default=2.0)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    
    parser.add_argument('--CB_beta', type=float, default=0.9)
    parser.add_argument('--CB_mode', type=str, default='average_w',
                    choices=['by_class', 'average_n', 'average_w', 'min_n'])

    parser.add_argument('--logit_neg_scale', type=float, default=2.0)
    parser.add_argument('--logit_init_bias', type=float, default=0.05)

    parser.add_argument('--map_alpha', type=float, default=0.1)
    parser.add_argument('--map_beta', type=float, default=10.0)
    parser.add_argument('--map_gamma', type=float, default=0.1)

    return parser.parse_args()