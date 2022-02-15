import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import os
import numpy as np
import random
from scipy import signal
from tqdm import tqdm
from torchvision.io import read_video, write_video, read_image
import soundfile as sf
import json
import matplotlib.pyplot as plt
import logging
import transforms as T

class DemoDataset(Dataset):
    """
    """

    def __init__(self,
                 video1_path,
                 video2_path,
                 align_point_1,
                 align_point_2,
                 span,
                 transform,
                 cut_type_filename= 'data/cut-type-train.json',
                 sampling='gaussian',
                 keep_audio=0, # 0 or 1 to keep the first or second video, -1 keep both
                 snippet_size=16,
                 seed=4165):
    
        self.video_paths = [video1_path, video2_path]
        self.alignment_points = [align_point_1, align_point_2]
        self.transform = transform
        self.sampling = sampling
        self.snippet_size = snippet_size
        self.span = span
        self.snippet_audio_size = None
        self.keep_audio = keep_audio
        self.seed = seed
        
        self.frames_1,self.audio_1 = None,None
        self.frames_2,self.audio_2 = None,None
        self.idx_pairs = None

        self.read_videos()
        self.create_pair_idxs_video()
        self.create_pair_idxs_audio()

        json_file = json.load(open(cut_type_filename))
        self.cut_types = json_file['cut_types']

    def read_videos(self):
        frames_1, audio_1, meta_1 = read_video(self.video_paths[0])
        frames_2, audio_2, meta_2  = read_video(self.video_paths[1])

        
        self.fps = meta_1['video_fps']
        self.audio_fps = meta_1['audio_fps']
        self.snippet_audio_size = self.snippet_size*(self.audio_fps/self.fps)
        self.frames_1 = frames_1[int(self.alignment_points[0]*self.fps):round((self.alignment_points[0]+self.span)*self.fps)]
        self.audio_1 = audio_1[:,int(self.alignment_points[0]*self.audio_fps):round((self.alignment_points[0]+self.span)*self.audio_fps)]
        
        self.frames_2 = frames_2[int(self.alignment_points[1]*self.fps):round((self.alignment_points[1]+self.span)*self.fps)]
        self.audio_2 = audio_2[:,int(self.alignment_points[1]*self.audio_fps):round((self.alignment_points[1]+self.span)*self.audio_fps)]
    
    def create_pair_idxs_video(self):
        total_frames = self.frames_1.shape[0]
        idxs = torch.arange(0,total_frames,self.snippet_size, dtype=torch.int32)
        idx_pairs_video = []
        for n0,n1,m1 in zip(idxs[:-2],idxs[1:-1],idxs[2:]):
            idx_pairs_video.append([[n0,n1],[n1,m1]])
        self.idx_pairs_video = torch.tensor(idx_pairs_video)
    
    def create_pair_idxs_audio(self):
        total_frames = self.audio_1.shape[1]
        idxs = torch.arange(0,total_frames,self.snippet_audio_size, dtype=torch.int32)
        idx_pairs_audio = []
        for n0,n1,m1 in zip(idxs[:-2],idxs[1:-1],idxs[2:]):
            idx_pairs_audio.append([[n0,n1],[n1,m1]])
        self.idx_pairs_audio = torch.tensor(idx_pairs_audio)
        
    def get_clip_spectogram(self, audio, samplerate, name=None):

        audio[audio > 1.] = 1.
        audio[audio < -1.] = -1.
        frequencies, times, spectrogram = signal.spectrogram(audio, samplerate, nperseg=512, noverlap=353)
        spectrogram = np.log(spectrogram+ 1e-7)
        mean = np.mean(spectrogram)
        std = np.std(spectrogram)
        spectrogram = np.divide(spectrogram-mean,std+1e-9)
        if name:
            plt.pcolormesh(times, frequencies, spectrogram, shading='gouraud')
            plt.savefig(name)
        return torch.from_numpy(spectrogram)

    def __getitem__(self, idx):
        this_video_idxs = self.idx_pairs_video[idx]
        video1 = self.frames_1[this_video_idxs[0][0]:this_video_idxs[0][1]]
        video2 = self.frames_2[this_video_idxs[1][0]:this_video_idxs[1][1]]

        out_video = self.transform(torch.cat((video1,video2),dim=0).permute(3,0,1,2))
        
        #output extended audio or output from both videos.

        this_audio_idxs = self.idx_pairs_audio[idx]
        if self.keep_audio == -1:
            audios = [self.audio1,self.audio2]
            kept_audio = audios[self.keep_audio]
            audio = kept_audio[:,this_audio_idxs[0][0]:this_audio_idxs[1][1]] #Pick the whole audio frames from the same video
            out_audio = self.get_clip_spectogram(audio, self.audio_fps)
        
        else:
            audio1 = self.audio_1[:,this_audio_idxs[0][0]:this_audio_idxs[0][1]]
            audio2 = self.audio_2[:,this_audio_idxs[1][0]:this_audio_idxs[1][1]]
            audios = [audio1,audio2]
            out_audio = self.get_clip_spectogram(torch.cat((audios[0],audios[1]),dim=1), self.audio_fps)            

        return out_video, out_audio
        
    def __len__(self):
        return len(self.idx_pairs_video)

def get_transforms():

    transform = torchvision.transforms.Compose(
        [
            T.ToTensorVideo(),
            T.Resize((128, 180)),
            T.NormalizeVideo(
                mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989)
            ),
            T.CenterCropVideo((112, 112)),
        ]
    )
    return transform

if __name__ == "__main__":

    transform = get_transforms()
    dataset =    DemoDataset(video1_path='/ibex/ai/home/pardogl/c2114/data/movies/demo/1A-3.mov',
                 video2_path='/ibex/ai/home/pardogl/c2114/data/movies/demo/1A-3.mov',
                 align_point_1=16,
                 align_point_2=40,
                 span=37,
                 transform=transform,
                 sampling='gaussian',
                 snippet_size=8, #per video side, in total snippet_size*2
                 seed=4165)
    
    dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            shuffle=False)
        
    for idx, data in enumerate(dataloader):
        print(f'batch idx {idx} - pair idx: {dataloader.dataset.idx_pairs_video[idx]}')
        print(f'video shape: {vid.shape}')
        print(f'audio shape: {audio.shape}')