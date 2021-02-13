import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import ffmpeg
import tqdm
import decord as de
de.bridge.set_bridge('torch')


class MovieDataset(Dataset):
    """Construct an untrimmed video classification dataset."""

    def __init__(self,
                 shots_filename,
                 videos_path = '../data/movies/youtube/',
                 num_positives_per_scene=10,
                 negative_positive_ratio=1,
                 snippet_size=16,
                 stride=8,
                 fps=24,
                 size=(112,112),
                 seed=424242):
    
        self.shots_df = pd.read_csv(shots_filename)
        self.shots_df_by_id = self.shots_df.groupby('video_id')
        self.videos_path = videos_path

        self.mode = 'train' if 'train' in shots_filename else 'val'
        self.snippet_size = snippet_size
        self.stride = stride
        self.fps = fps
        self.size = size
        self.seed = seed
        self.num_positives_per_scene = num_positives_per_scene
        self.negative_positive_ratio = negative_positive_ratio

        self.candidates = None
        self.video_names = None
        self.set_candidates()

    def get_average_shots_per_scene(self):
        num_shots = 0
        for name, df in self.shots_df_by_id:
            num_shots += len(df)
        avg_num_shots = num_shots/len(self.shots_df_by_id)
        return avg_num_shots

    def get_full_clip(self, video_path):
        vr = de.VideoReader(video_path, width=self.size[1], height=self.size[0])
        return vr
    
    def cut_clip_decord(self, vr, start_time, frame_span=24):
        fps = round(vr.get_avg_fps())
        start_frame = int(np.ceil(start_time * self.fps))
        end_frame = start_frame + frame_span
        frames = vr.get_batch(range(start_frame, end_frame, 1))
        return frames.type(torch.float32) 

    def get_clip_decord(self, video_path, start_time, time_span=1):
        ctx = de.gpu(0)

        vr = de.VideoReader(video_path, width=self.size[1], height=self.size[0])
        fps = round(vr.get_avg_fps())
        start_frame = int(np.ceil(start_time*fps))
        end_frame = int(np.floor(start_frame + fps*time_span))
        frames = vr.get_batch(range(start_frame, end_frame, 1))

        return frames

    def __len__(self):
        return (len(self.candidates_per_video))
    
    # def __getitem__(self, idx):
    #     video_path = f'{self.videos_path}/{self.candidates[idx][0]}/{self.candidates[idx][0]}.mp4'
    #     start_time = self.candidates[idx][3]-0.5
    #     end_time = self.candidates[idx][4]
    #     label = self.labels[idx]

    #     if os.path.isfile(video_path):
            
    #         if label == 1:

    #             video = self.get_clip_decord(video_path, start_time)
    #             video = video.permute(3, 0, 1, 2)

    #         if label == 0:

    #             video_left = self.get_clip_decord(video_path, start_time, time_span=0.5)
    #             video_left = video_left.permute(3, 0, 1, 2)
    #             video_right = self.get_clip_decord(video_path, end_time, time_span=0.5)
    #             video_right = video_right.permute(3, 0, 1, 2)

    #             video = torch.cat((video_left,video_right), dim=1)
                
                
    #     else:
    #         print(f'{video_path} video does not exist')
    #         video = torch.zeros(1)
        
    #     video = video.type(torch.FloatTensor)
    #     return video, label

    def __getitem__(self, idx):
        name = self.video_names[idx]
        video_path = f'{self.videos_path}/{name}/{name}.mp4'
        
        video_frames = self.get_full_clip(video_path)

        this_candidates = self.candidates_per_video[name]

        total_samples = torch.empty((0,self.fps,self.size[0],self.size[1],3))
        total_labels = torch.empty((0), dtype=torch.uint8)
        for this_candidate, this_label in zip(this_candidates['candidates'], this_candidates['labels']):

            if this_label == 1:
                this_start_time = this_candidate[2]- 0.5
                this_frames = self.cut_clip_decord(video_frames, this_start_time, frame_span=self.fps)
            if this_label == 0:
                start_time_left = this_candidate[2]
                this_frames_left = self.cut_clip_decord(video_frames, start_time_left-0.5, frame_span=int(0.5*self.fps))
                start_time_right = this_candidate[2]
                this_frames_right = self.cut_clip_decord(video_frames, start_time_right, frame_span=int(0.5*self.fps))
                this_frames = torch.cat((this_frames_left,this_frames_right), dim=0)

            total_samples = torch.cat((total_samples, this_frames.unsqueeze(0)), dim=0)
            total_labels = torch.cat((total_labels, torch.tensor([this_label],dtype=torch.uint8)), dim=0)

        return name, total_samples, total_labels
    
    def collate_fn(self, data_lst):
        video_names = [_video_name for _video_name, _, _ in data_lst]
        samples = torch.cat([_sample for _, _sample, _ in data_lst],dim=0)
        labels = torch.cat([_label for _, _, _label in data_lst],dim=0)

        samples = samples.permute(0, 4, 1, 2, 3)
        return video_names, samples, labels

    def set_candidates(self):
        print(f'Setting candidates for {self.mode}')
        self.candidates = []
        self.labels = []
        self.candidates_per_video = {}
        for name, df in self.shots_df_by_id:
            candidates = []
            labels = []
            self.candidates_per_video[name] = {}
            cand_counter = 0
            while cand_counter < self.num_positives_per_scene:
                cand_counter += 1
                # Find positive candidates
                candidates.append(df.sample().values.tolist()[0])
                labels.append(1)
                # Find negative candidates
                for i in range(self.negative_positive_ratio):    
                    negative = df.sample().values.tolist()[0][0:3]
                    negative.extend(df.sample().values.tolist()[0][3:])
                    candidates.append(negative)
                    labels.append(0)
            self.candidates_per_video[name]['candidates'] = candidates
            self.candidates_per_video[name]['labels'] = labels
        
        self.video_names = list(self.candidates_per_video.keys())