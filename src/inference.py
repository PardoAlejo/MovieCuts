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

def sigmoid(X):
    return 1/(1+torch.exp(-X.squeeze()))


def get_dataloader():
    
    transform = get_transforms()
    dataset =    DemoDataset(video2_path='/ibex/ai/home/pardogl/c2114/data/movies/demo/1A-3.mov', #1B-1.mov', #1A-3.mov', #
                 video1_path='/ibex/ai/home/pardogl/c2114/data/movies/demo/1B-1.mov', #1A-3.mov', # 1B-1.mov', #
                 align_point_2=41.07, #17.04, #41.07, # 17.04,
                 align_point_1=18.04, #41.07, #17.04, # 41.07, #
                 span=36,
                 transform=transform,
                 sampling='gaussian',
                 keep_audio=1, # 0 or 1 to keep the first or second video
                 snippet_size=8, #per video side, in total snippet_size*2
                 seed=4165)
    
    dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
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

    dataloader = get_dataloader()
    headers = ['Start','End']+dataloader.dataset.cut_types
    rows = []
    for idx, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        vid,audio = data
        this_time_stamp = dataloader.dataset.idx_pairs_video[idx][0]
        logits, _, _ = model(vid,audio)
        rows.append(this_time_stamp.cpu().tolist()+sigmoid(logits[0].detach()).cpu().tolist())
    
    results_df = pd.DataFrame(rows, columns=headers)
    results_df.to_csv('OUTPUTS/demo_audio2_2_1-18.csv',index=False)
