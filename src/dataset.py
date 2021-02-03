import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import ffmpeg
import tqdm

class MovieDataset(Dataset):
    """Construct an untrimmed video classification dataset."""

    def __init__(self,
                 shots_filename,
                 videos_path = '../data/movies/youtube/',
                 num_positives_per_scene=10,
                 negative_positive_ratio=1,
                 snippet_size=16,
                 stride = 8,
                 fps=24,
                 size=(112,199),
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
        self.set_candidates()

    def get_average_shots_per_scene(self):
        num_shots = 0
        for name, df in self.shots_df_by_id:
            num_shots += len(df)
        avg_num_shots = num_shots/len(self.shots_df_by_id)
        return avg_num_shots

    def _get_video_dim(self, video_path):
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams']
                             if stream['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        return height, width

    def __len__(self):
        return (len(self.candidates))
    
    def __getitem__(self, idx):
        video_path = f'{self.videos_path}/{self.candidates[idx][0]}/{self.candidates[idx][0]}.mp4'
        start_time = self.candidates[idx][3]-0.5
        end_time = self.candidates[idx][4]+0.5
        label = self.labels[idx]

        if os.path.isfile(video_path):
            
            height, width = self.size

            if label == 1:
                cmd = (
                    ffmpeg
                    .input(video_path, ss=start_time)
                    .filter('fps', fps=self.fps)
                    .filter('scale', width, height)
                    )
                out, _ = (
                cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24', vframes=self.snippet_size)
                .run(capture_stdout=True, quiet=True)
                )

                video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
                video = torch.from_numpy(video.astype('float32'))
                video = video.permute(3, 0, 1, 2)

            if label == 0:
                cmd = (
                    ffmpeg
                    .input(video_path, ss=start_time)
                    .filter('fps', fps=self.fps)
                    .filter('scale', width, height)
                    )
                out, _ = (
                cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24', vframes=int(self.snippet_size/2))
                .run(capture_stdout=True, quiet=True)
                )

                video_left = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
                video_left = torch.from_numpy(video_left.astype('float32'))
                video_left = video_left.permute(3, 0, 1, 2)

                cmd = (
                    ffmpeg
                    .input(video_path, ss=end_time)
                    .filter('fps', fps=self.fps)
                    .filter('scale', width, height)
                    )
                out, _ = (
                cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24', vframes=int(self.snippet_size/2))
                .run(capture_stdout=True, quiet=True)
                )

                video_right = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
                video_right = torch.from_numpy(video_right.astype('float32'))
                video_right = video_right.permute(3, 0, 1, 2)

                video = torch.cat((video_left,video_right), dim=1)
                
        else:
            print(f'{video_path} video does not exist')
            video = torch.zeros(1)
        return video, label

    def set_candidates(self):
        print(f'Setting candidates for {self.mode}')
        self.candidates = []
        self.labels = []
        for name, df in self.shots_df_by_id:
            cand_counter = 0
            while cand_counter < self.num_positives_per_scene:
                cand_counter += 1
                # Find positive candidates
                self.candidates.append(df.sample().values.tolist()[0])
                self.labels.append(1)

                # Find negative candidates
                for i in range(self.negative_positive_ratio):    
                    negative = df.sample().values.tolist()[0][0:3]
                    negative.extend(df.sample().values.tolist()[0][3:])
                    self.candidates.append(negative)
                    self.labels.append(0)
