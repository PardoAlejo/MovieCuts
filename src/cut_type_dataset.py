import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import random
import ffmpeg
from PIL import Image
from scipy import signal
from tqdm import tqdm
from torchvision.io import read_video, write_video
import soundfile as sf
import json
import matplotlib.pyplot as plt

class CutTypeDataset(Dataset):
    """Construct an untrimmed video classification dataset.
    stream: visual, audio, audiovisual
    time_audio_span: Duration of the audio window in seconds
    """

    def __init__(self,
                 shots_filenames,
                 cut_type_filename,
                 visual_stream=True,
                 audio_stream=True,
                 transform=None,
                 videos_path = 'data/framed_clips',
                 cache_path = './.cache',
                 augment_temporal_shift=True,
                 pos_delta_range=list(range(5)),
                 augment_spatial_flip=True,
                 sampling='gaussian',
                 snippet_size=16,
                 time_audio_span=10,
                 data_percent=0.1,
                 seed=4165):
        
        self.seed = seed
        self.mode = 'train' if 'train' in cut_type_filename else 'val' if 'val' in cut_type_filename else 'test'

        dataframes = []
        for filename in shots_filenames:
            this_shots_df = pd.read_csv(filename)
            dataframes.append(this_shots_df)
        self.shots_df = pd.concat(dataframes)
        self.shots_df_by_video_id = self.shots_df.groupby('video_id')
        self.shots_df_by_movie_id = self.shots_df.groupby('movie_id')
        self.videos_path = videos_path
        self.cache_path = cache_path
        
        self.data_percent = data_percent

        self.transform = transform

        json_file = json.load(open(cut_type_filename))
        self.cut_type_annotations = json_file['annotations']

        self.cut_types = json_file['cut_types']
        self.clip_names = list(self.cut_type_annotations.keys())
        if self.mode == 'train':
            num_samples = int(data_percent*len(self.clip_names))
            self.clip_names = random.sample(self.clip_names, k=num_samples)

        self.visual_stream = visual_stream
        self.audio_stream = audio_stream
        self.snippet_size = snippet_size
        self.time_audio_span = time_audio_span

        if sampling=='gaussian':
            self.sampling_function = self.generate_gaussian_sampling
        elif sampling=='uniform':
            self.sampling_function = self.generate_uniform_sampling
        elif sampling=='fixed':
            self.sampling_function = self.generate_fix_window_sampling

        self.augment_spatial_flip = augment_spatial_flip
        self.augment_temporal_shift = augment_temporal_shift
        self.pos_delta_range = pos_delta_range

        self.candidates = None
        self.clips_to_fps = dict(zip(self.shots_df.clip_id.tolist(),self.shots_df.fps.tolist()))
        
        if self.mode == 'train':
            self.cache_filename = f'{self.cache_path}/candidates_{self.mode}_cut_type_percent_{int(data_percent*100)}.json'
        else:
            self.cache_filename = f'{self.cache_path}/candidates_{self.mode}_cut_type_percent_{int(1*100)}.json'
        if not os.path.exists(self.cache_filename):
            self.set_candidates()
        else:
            print(f'Cache file found at: {self.cache_filename}')
            self.read_cache_candidates()

    def __len__(self):
        return (len(self.clip_names))

    def get_average_shots_per_scene(self):
        num_shots = 0
        for name, df in self.shots_df_by_video_id:
            num_shots += len(df)
        avg_num_shots = num_shots/len(self.shots_df_by_video_id)
        return avg_num_shots
    
    def get_clip_audio(self, clip_path):
        clip_name = os.path.basename(clip_path)
        audio_path = f'{clip_path}/{clip_name}.wav'
        audio, samplerate = sf.read(audio_path)
        span_audio = int(self.time_audio_span*samplerate)

        if len(audio) < span_audio:
            pad_ratio = span_audio - len(audio)
            audio = np.pad(audio, (int(pad_ratio/2), int(pad_ratio/2)),constant_values=(audio[0], audio[-1]))
        elif len(audio) > span_audio:
            cut_ratio = int((len(audio) - span_audio)/2)
            audio = audio[cut_ratio:-cut_ratio]

        return audio[0:span_audio], samplerate

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

    def get_clip_from_frames(self, clip_path, cut_time, window, fps):
        """
        clip_path path to clip frames
        """
        cut_id = int(cut_time*fps)
        ids = self.sampling_function(cut_id, window)

        filenames = [f'{clip_path}/frames/{i:06d}.jpg' for i in ids]
        frames = []

        for f in filenames:
            img = Image.open(f)
            img = img.convert('RGB')
            frames.append(img)
            
        if (self.augment_spatial_flip) & (bool(random.getrandbits(1))):
            frames = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in frames]

        frames = [torch.from_numpy(np.asarray(img).copy()) for img in frames]
        frames = torch.stack(frames, 0)

        return frames
    
    def generate_gaussian_sampling(self, cut_frame_id, window):
        ids = [int(random.gauss(cut_frame_id, window/3)) for _ in range(self.snippet_size)]
        ids = [x if x>0 else 0 for x in ids]
        ids = [x if x<cut_frame_id+window else cut_frame_id+window for x in ids]
        ids.sort()
        return ids
    
    def generate_unfinorm_sampling(self, cut_frame_id, window):
        low = cut_frame_id - window
        high = cut_frame_id + window
        ids = [int(random.uniform(low, high)) for _ in range(self.snippet_size)]
        ids.sort()
        return ids

    def generate_fix_window_sampling(self, cut_frame_id, window):
        low = cut_frame_id - int(window/2)
        high = cut_frame_id + int(window/2)
        ids = list(np.linspace(low, self.snippet_size, high).astype(np.uint8))
        return ids

    def __getitem__(self, idx):
        
        if self.augment_temporal_shift:
            pos_delta = np.random.choice(self.pos_delta_range)
        else:
            pos_delta = 0 # exact cut -- left/right shots see same  # frames.
        
        clip_name = self.clip_names[idx]
        clip_path = f'{self.videos_path}/{clip_name}'
        labels = torch.tensor(self.candidates[clip_name]['labels'])
        shot_times = self.candidates[clip_name]['shot_times']
        fps = self.clips_to_fps[clip_name]
        cut_time = shot_times[1] + pos_delta/fps
        end_time = shot_times[2]
        window_time = min(cut_time, end_time-cut_time)
        window_frames = int(window_time*fps)
        
        if self.visual_stream:
            clip = self.get_clip_from_frames(clip_path, cut_time, window_frames, fps)
            if self.transform:
                clip = self.transform(clip)

        if self.audio_stream:
            audio, rate = self.get_clip_audio(clip_path)
            spectogram = self.get_clip_spectogram(audio, rate)            
        

        if self.visual_stream and not self.audio_stream:
            return clip, labels, clip_name
        
        elif not self.visual_stream and self.audio_stream:
            return spectogram.unsqueeze(0).float(), labels, clip_name
        
        elif self.visual_stream and self.audio_stream:
            return clip, spectogram.unsqueeze(0).float(), labels, clip_name


    def read_cache_candidates(self):
        self.candidates = json.load(open(self.cache_filename))
        print('Candidates readed from cache file')

    def set_candidates(self):
        print(f'Setting candidates for {self.mode}')
        self.candidates = {}

        for clip_name in tqdm(self.clip_names):
            row = self.shots_df[self.shots_df.clip_id==clip_name].iloc[0]
            cut_time = (row.shot_left_end + row.shot_right_start)/2
            shot_times = [row.shot_left_start, cut_time, row.shot_right_end]
            # Center around zero since annotations come from original scenes
            shot_times = [x-row.shot_left_start for x in shot_times]
            self.candidates[clip_name] = {'shot_times':shot_times, 'labels':self.cut_type_annotations[clip_name]['labels']}

        print(f'Saving cache candidates file in: {self.cache_filename}')
        with open(self.cache_filename, 'w') as f:
            json.dump(self.candidates, f)

