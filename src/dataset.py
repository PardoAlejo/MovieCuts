import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import ffmpeg
import tqdm
from torchvision.io import read_video, write_video

class MovieDataset(Dataset):
    """Construct an untrimmed video classification dataset."""

    def __init__(self,
                 shots_filename,
                 transform=None,
                 videos_path = '/tmp/youtube', #'../data/movies/youtube', 
                 negative_positive_ratio=1,
                 augment_temporal_shift=True,
                 pos_delta_range=list(range(5)),
                 snippet_size=16,
                 original_fps=24,
                 network_fps=15,   
                 size=(112, 112),
                 seed=424242):
    
        self.shots_df = pd.read_csv(shots_filename)
        self.shots_df_by_video_id = self.shots_df.groupby('video_id')
        self.shots_df_by_movie_id = self.shots_df.groupby('movie_id')
        self.videos_path = videos_path

        self.transform = transform

        self.mode = 'train' if 'train' in shots_filename else 'val'
        self.snippet_size = snippet_size
        self.original_fps = original_fps
        self.network_fps = network_fps
        self.time_span = self.snippet_size/self.network_fps
        self.height, self.width = size
        self.seed = seed
        self.negative_positive_ratio = negative_positive_ratio

        self.augment_temporal_shift = augment_temporal_shift
        self.pos_delta_range = pos_delta_range

        self.candidates = None
        self.set_candidates()

    def get_average_shots_per_scene(self):
        num_shots = 0
        for name, df in self.shots_df_by_video_id:
            num_shots += len(df)
        avg_num_shots = num_shots/len(self.shots_df_by_video_id)
        return avg_num_shots

    def get_clip_ffmpeg(self, video_path, start_time, time_span):
        vframes = int(np.floor(time_span*self.original_fps))
        cmd = (
            ffmpeg
            .input(video_path, ss=start_time)
            .filter('fps', fps=self.original_fps)
            .filter('scale', 1280, 720)
            )
        out, _ = (
        cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24', vframes=vframes)
        .run(capture_stdout=True, quiet=True)
        )

        clip = np.frombuffer(out, np.uint8).reshape([-1, 720, 1280, 3])
        clip = torch.from_numpy(clip.astype('uint8'))

        return clip

    def get_clip_pytorch(self, video_path, start_time, time_span):
    
        clip, _, info  = read_video(video_path, start_pts=start_time, end_pts=start_time+time_span, pts_unit='sec')
        
        fps = info['video_fps']
        
        return clip

    def __len__(self):
        return (len(self.candidates))
    
    def resample_video_idx(self, num_frames, original_fps, new_fps):
        step = float(original_fps) / new_fps
        if step.is_integer():
            # optimization: if step is integer, don't need to perform
            # advanced indexing
            step = int(step)
            return slice(None, None, step)
        idxs = torch.arange(num_frames-1, dtype=torch.float32) * step
        idxs = idxs.floor().to(torch.int64)
        
        return idxs

    def __getitem__(self, idx):
        video_path = f'{self.videos_path}/{self.candidates[idx][0]}/{self.candidates[idx][0]}.mp4'
        
        if self.augment_temporal_shift:
            pos_delta = np.random.choice(self.pos_delta_range)
        else:
            pos_delta = 0 # exact cut -- left/right shots see same  # frames.

        left_duration = self.time_span*0.5 - pos_delta/self.original_fps
        start_time = self.candidates[idx][2] - left_duration

        right_duration = self.time_span*0.5 + pos_delta/self.original_fps
        end_time = self.candidates[idx][3] + right_duration
        
        label = self.labels[idx]
        
        if os.path.isfile(video_path):
            
            if label == 1:

                clip = self.get_clip_ffmpeg(video_path, start_time, time_span=self.time_span)

            if label == 0:
                clip_left = self.get_clip_ffmpeg(video_path, start_time, time_span=left_duration)
                clip_right = self.get_clip_ffmpeg(video_path, end_time, time_span=right_duration)

                clip = torch.cat((clip_left, clip_right), dim=0)
                 
                
        else:
            print(f'{video_path} video does not exist')
            clip = torch.zeros(1)

        idxs = self.resample_video_idx(self.snippet_size, self.original_fps, self.network_fps)
        # write_video(f'examples_ffmpeg/{label}/{self.candidates[idx][0]}.mp4',clip[idxs,:,:,:],self.network_fps)
        if self.transform:
            clip = self.transform(clip)
            
        return clip[:,idxs,:,:], label

    def set_candidates(self):
        print(f'Setting candidates for {self.mode}')
        self.candidates = []
        self.labels = []
        for idx, row in tqdm.tqdm(self.shots_df.iterrows(),total=len(self.shots_df)):
            this_candidates = []
            this_labels = []
            
            # Find positive candidates
            df = self.shots_df[self.shots_df.video_id==row.video_id]
            this_candidates.append(df.sample().values.tolist()[0])
            this_labels.append(1)
            # Find negative candidates
            for i in range(self.negative_positive_ratio):  
                left_found = False
                while not left_found:
                    left = df.sample().values.tolist()[0]
                    if left[2] - left[1] > 1:
                        negative = left[0:3]
                        left_found = True
                right_found = False
                while not right_found:
                    right = df.sample().values.tolist()[0]
                    if right[2] - right[1] > 1:
                        negative.extend(right[1:3])
                        right_found = True
                negative.extend(right[5:])
                this_candidates.append(negative)
                this_labels.append(0)
            self.candidates.extend(this_candidates)
            self.labels.extend(this_labels)