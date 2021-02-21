import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import ffmpeg
import tqdm
from torchvision.io import write_video

class MovieDataset(Dataset):
    """Construct an untrimmed video classification dataset."""

    def __init__(self,
                 shots_filename,
                 transform=None,
                 videos_path = '/tmp/youtube',
                 num_positives_per_scene=5,
                 negative_positive_ratio=1,
                 across_scene_negs=False,
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
        self.across_scene_negs = across_scene_negs
        self.snippet_size = snippet_size
        self.original_fps = original_fps
        self.network_fps = network_fps
        self.time_span = self.snippet_size/self.network_fps
        self.height, self.width = size
        self.seed = seed
        self.num_positives_per_scene = num_positives_per_scene
        self.negative_positive_ratio = negative_positive_ratio

        self.candidates = None
        self.video_names = None
        self.set_candidates()

    def get_average_shots_per_scene(self):
        num_shots = 0
        for name, df in self.shots_df_by_video_id:
            num_shots += len(df)
        avg_num_shots = num_shots/len(self.shots_df_by_video_id)
        return avg_num_shots

    def get_clip_ffmpeg(self, video_path, start_time, time_span):
        vframes = int(time_span*self.original_fps)
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
        start_time = self.candidates[idx][3]-self.time_span*0.5
        end_time = self.candidates[idx][4]
        label = self.labels[idx]
        
        if os.path.isfile(video_path):
            
            if label == 1:

                clip = self.get_clip_ffmpeg(video_path, start_time, time_span=self.time_span)

            if label == 0:
                if self.across_scene_negs:
                    left_path = f'{self.videos_path}/{self.candidates[idx][0]}/{self.candidates[idx][0]}.mp4'
                    clip_left = self.get_clip_ffmpeg(left_path, start_time, time_span=self.time_span*0.5)
                    rigth_path = f'{self.videos_path}/{self.candidates[idx][-2]}/{self.candidates[idx][-2]}.mp4'
                    clip_right = self.get_clip_ffmpeg(rigth_path, end_time, time_span=self.time_span*0.5)
                else:
                    clip_left = self.get_clip_ffmpeg(video_path, start_time, time_span=self.time_span*0.5)
                    clip_right = self.get_clip_ffmpeg(video_path, end_time, time_span=self.time_span*0.5)

                clip = torch.cat((clip_left, clip_right), dim=0)
                 
                
        else:
            print(f'{video_path} video does not exist')
            clip = torch.zeros(1)

        idxs = self.resample_video_idx(self.snippet_size, self.original_fps, self.network_fps)
        
        if self.transform:
            clip = self.transform(clip)
            
        return clip[:,idxs,:,:], label

    def set_candidates(self):
        print(f'Setting candidates for {self.mode}')
        self.candidates = []
        self.labels = []
        self.candidates_per_video = {}
        for name, df in self.shots_df_by_video_id:
            this_candidates = []
            this_labels = []
            self.candidates_per_video[name] = {}
            cand_counter = 0
            while cand_counter < self.num_positives_per_scene:
                cand_counter += 1
                # Find positive candidates
                this_candidates.append(df.sample().values.tolist()[0])
                this_labels.append(1)
                # Find negative candidates
                for i in range(self.negative_positive_ratio):  
                    negative = df.sample().values.tolist()[0]
                    movie_id = negative[-1]
                    negative = negative[0:3]
                    if self.across_scene_negs:
                        # get a sample of a random scene of the same movie
                        right = self.shots_df[self.shots_df.movie_id == movie_id].sample()
                        vid_id = right.video_id.values[0]
                        # Store the scene id to read from
                        right_vid = right.values.tolist()[0]
                        right_vid[-2] = vid_id
                        negative.extend(right_vid[3:])
                    else:
                        negative.extend(df.sample().values.tolist()[0][3:])
                    this_candidates.append(negative)
                    this_labels.append(0)
            self.candidates.extend(this_candidates)
            self.labels.extend(this_labels)
            self.candidates_per_video[name]['candidates'] = this_candidates
            self.candidates_per_video[name]['labels'] = this_labels
        
        self.video_names = list(self.candidates_per_video.keys())