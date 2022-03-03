import numpy as np
import torch
import torchvision
import sys
import os
from tqdm import tqdm
sys.path.insert(1, f'{os.getcwd()}/models')
sys.path.insert(1, f'{os.getcwd()}/utils')
from config import config
from video_resnet import r2plus1d_18
from audio_model import AVENet
from audio_visual_model import AudioVisualModel
from demo_dataset import DemoDataset, get_transforms
from torch.utils.data import DataLoader
import transforms as T
import logging 
from collections import OrderedDict
logging.basicConfig(level=logging.DEBUG)
import pandas as pd
from itertools import cycle

def sigmoid(X):
    return 1/(1+torch.exp(-X.squeeze()))


def get_dataloader(video1_path, video2_path,align_point_1,align_point_2,
                    span,transform,keep_audio,snippet_size,candidate_window,seed=4165):
    
    print(video1_path, video2_path,align_point_1,align_point_2,span)
    dataset =    DemoDataset(
                video1_path      = video1_path,
                video2_path      = video2_path,
                align_point_1    = align_point_1,
                align_point_2    = align_point_2,
                span             = span,
                transform        = transform,
                keep_audio       = keep_audio,
                snippet_size     = snippet_size,
                candidate_window = candidate_window,
                seed             = seed)
    
    dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=4,
            pin_memory=True,
            shuffle=False)

    return dataloader



if __name__ == "__main__":

    model_path = 'checkpoints/epoch=7_Validation_loss=1.91.ckpt'
    model = AudioVisualModel(num_classes=10)
    checkpoint = torch.load(model_path)
    
    new_state_dict = OrderedDict()
    for name,params in checkpoint['state_dict'].items():
        new_state_dict[name.replace('audio_visual_network.','')] = params
    model.load_state_dict(new_state_dict)

    base_path = '/ibex/ai/home/pardogl/LTC-e2e/User_Study'
    info_df = pd.read_csv(f'{base_path}/video_info.csv')
    
    # video_paths = ['/ibex/ai/home/pardogl/c2114/data/movies/demo/1A-3.mov',
    #                 '/ibex/ai/home/pardogl/c2114/data/movies/demo/1B-1.mov']
    # align_points = [41.07,16.05]
    for idx, row in info_df.iterrows():
        folder = f'demo_{row.Demo:02d}'
    
        # video_paths = ['/ibex/ai/home/pardogl/c2114/data/movies/demo/21A_2.mov',
        #                 '/ibex/ai/home/pardogl/c2114/data/movies/demo/21B_3.mov']
        # align_points = [16.0,13.0]
        video_paths = [f'{base_path}/data/{folder}/{row.Clip1}',
                       f'{base_path}/data/{folder}/{row.Clip2}']
        align_points = [row.Offset1,row.Offset2]
        span = [row.Span]
        keep_audio = [0]
        snippet_size = [16]
        candidate_window = [0.5]
        
        transform = get_transforms()
        for audio_channel in keep_audio:
            for i in range(2):
                out_name = f'User_Study/outputs/{folder}_audio_{audio_channel}_aligning_{candidate_window[0]}_{i}.csv'
                if os.path.exists(out_name):
                    print(f'{out_name} already exists')
                    continue
                dataloader = get_dataloader(
                            video1_path      = video_paths[0],
                            video2_path      = video_paths[1],
                            align_point_1    = align_points[0],
                            align_point_2    = align_points[1],
                            span             = span[0],
                            transform        = transform,
                            keep_audio       = audio_channel,
                            snippet_size     = snippet_size[0],
                            candidate_window = candidate_window[0]
                            )
                
                headers = ['Start1','Start2','ClipSize']+dataloader.dataset.cut_types
                rows = []
                for idx, data in tqdm(enumerate(dataloader), total=len(dataloader)):
                    vid,audio = data
                    this_time_stamp = dataloader.dataset.idx_pairs_video[idx]
                    logits, _, _ = model(vid,audio)
                    rows.append(this_time_stamp.cpu().tolist() + [dataloader.dataset.clip_size] + sigmoid(logits[0].detach()).cpu().tolist())
                
                results_df = pd.DataFrame(rows, columns=headers)
                results_df.to_csv(out_name,index=False)

                # Flip order and do it again
                video_paths = video_paths[::-1]
                align_points = align_points[::-1]