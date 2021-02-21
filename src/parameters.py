import argparse


def get_params():
    parser = argparse.ArgumentParser(description='Easy video feature extractor')
        
    parser.add_argument('--shots_file_name_train', type=str, default='../data/used_cuts_train.csv',
                        help='Shots for training')
    parser.add_argument('--shots_file_name_val', type=str, default='../data/used_cuts_val.csv',
                        help='Shots for validation')

    # Reading arguments
    parser.add_argument("--scale_h", default=128, type=int,
                        help="Scale H to read")
    parser.add_argument("--scale_w", default=180, type=int,
                        help="Scale H to read")
    parser.add_argument("--crop_size", default=112, type=int, 
                        help="number of frames per clip")


    parser.add_argument('--candidates_per_scene', type=int, default=10,
                        help='Number of candidates per scene')
    parser.add_argument('--negative_positive_ratio_val', type=int, default=5,
                        help='Ratio for negatives:positives for validation')
    parser.add_argument('--across_scene_negs', action='store_true',
                        help='Take across scene negatives or not, default same scene')

    

    # Batch Size and initial learning rates
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers for data loading')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument("--initial_lr", default=0.001, type=float, 
                        help="initial learning rate")
    parser.add_argument("--fc_lr", default=0.001, type=float, 
                        help="fully connected learning rate")

    # Scheduler parameters
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="momentum")
    parser.add_argument('--lr_decay', type=float, default=0.9,
                        help='Lr decay')
    parser.add_argument("--weight-decay", default=1e-4, type=float, 
                        help="weight decay (default: 1e-4)")
    parser.add_argument("--lr-milestones",nargs="+",default=[4, 8],
                        type=int,help="decrease lr on milestones")
    parser.add_argument("--lr-gamma",default=0.1,type=float,
                        help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-warmup-epochs", default=2, type=int,
                        help="number of warmup epochs")

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

    # Test or load checkpoint
    parser.add_argument('--checkpoint', type=str, default='store_true',
                        help='Checkpoint to test or resume')
    parser.add_argument('--test', action='store_true',
                        help='Checkpoint to test or resume')                    

    return parser.parse_args()