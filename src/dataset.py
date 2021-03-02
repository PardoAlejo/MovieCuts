import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import random
import ffmpeg
from PIL import Image
from scipy import signal
import tqdm
from torchvision.io import read_video, write_video
import soundfile as sf
import json
import matplotlib.pyplot as plt

class MovieDataset(Dataset):
    """Construct an untrimmed video classification dataset.
    stream: visual, audio, audiovisual
    """

    def __init__(self,
                 shots_filename,
                 visual_stream=True,
                 audio_stream=True,
                 transform=None,
                 videos_path = 'data/framed_clips',
                 cache_path = './.cache',
                 negative_positive_ratio=1,
                 augment_temporal_shift=True,
                 pos_delta_range=list(range(5)),
                 augment_spatial_flip=True,
                 snippet_size=16,
                 original_fps=24,
                 network_fps=15,   
                 size=(112, 112),
                 seed=4165):
    
        self.shots_df = pd.read_csv(shots_filename)
        self.shots_df_by_video_id = self.shots_df.groupby('video_id')
        self.shots_df_by_movie_id = self.shots_df.groupby('movie_id')
        self.videos_path = videos_path
        self.cache_path = cache_path

        self.transform = transform

        self.mode = 'train' if 'train' in shots_filename else 'val'
        self.visual_stream = visual_stream
        self.audio_stream = audio_stream
        self.snippet_size = snippet_size
        self.original_fps = original_fps
        self.network_fps = network_fps
        self.time_span = self.snippet_size/self.network_fps
        self.height, self.width = size
        self.seed = seed
        self.negative_positive_ratio = negative_positive_ratio

        self.augment_spatial_flip = augment_spatial_flip
        self.augment_temporal_shift = augment_temporal_shift
        self.pos_delta_range = pos_delta_range

        self.candidates = None
        self.clips_to_fps = dict(zip(self.shots_df.clip_id.tolist(),self.shots_df.fps.tolist()))
        
        shot_file_base_name = os.path.basename(shots_filename).replace('.csv','')
        self.cache_filename = f'{self.cache_path}/{shot_file_base_name}_np-ratio_{self.negative_positive_ratio}_seed_{self.seed}.json'

        if not os.path.exists(self.cache_filename):
            self.set_candidates()
        else:
            print(f'Cache file found at: {self.cache_filename}')
            self.read_cache_candidates()

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
    
    def get_clip_audio(self, clip_path, start_time, time_span):
        clip_name = os.path.basename(clip_path)
        audio_path = f'{clip_path}/{clip_name}.wav'
        data, samplerate = sf.read(audio_path)
        start_sample = int(start_time*samplerate)
        end_sample = start_sample + int(time_span*samplerate)

        audio = data[start_sample:end_sample]
        return audio, samplerate

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

    def get_clip_from_frames(self, clip_path, start_time, time_span, fps):
        """
        clip_path path to clip frames
        """
        
        start_frame = int(start_time*fps)
        vframes = int(np.ceil(time_span*fps))

        filenames = [f'{clip_path}/frames/{i:06d}.jpg' for i in range(start_frame, start_frame+vframes, 1)]
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
        
        if self.augment_temporal_shift:
            pos_delta = np.random.choice(self.pos_delta_range)
        else:
            pos_delta = 0 # exact cut -- left/right shots see same  # frames.

        label = self.labels[idx]
                    
        if label == 1:
            clip_name = self.candidates[idx][0]
            fps = self.clips_to_fps[clip_name]
            left_duration = self.time_span*0.5 + pos_delta/fps
            start_time = self.candidates[idx][2] - left_duration
            clip_path = f'{self.videos_path}/{clip_name}'
            if self.visual_stream:
                clip = self.get_clip_from_frames(clip_path, start_time, self.time_span, fps)
                idxs = self.resample_video_idx(self.snippet_size, fps, self.network_fps)
            if self.audio_stream:
                audio, rate = self.get_clip_audio(clip_path, start_time, self.time_span)
                spectogram = self.get_clip_spectogram(audio, rate)

        if label == 0:
            clip_name_left = self.candidates[idx][0]
            clip_path_left = f'{self.videos_path}/{clip_name_left}'
            fps_left = self.clips_to_fps[clip_name_left]
            left_duration = self.time_span*0.5 + pos_delta/fps_left
            start_time = self.candidates[idx][2] - left_duration

            clip_name_right = self.candidates[idx][3]
            clip_path_right = f'{self.videos_path}/{clip_name_right}'
            fps_right = self.clips_to_fps[clip_name_right]
            right_duration = self.time_span*0.5 - pos_delta/fps_right
            end_time = self.candidates[idx][4]

            if self.visual_stream:
                clip_left = self.get_clip_from_frames(clip_path_left, start_time, left_duration, fps_left)
                clip_right = self.get_clip_from_frames(clip_path_right, end_time, right_duration, fps_right)

                clip = torch.cat((clip_left, clip_right), dim=0)
                idxs = self.resample_video_idx(self.snippet_size, fps_left, self.network_fps)

            if self.audio_stream:
                audio_left, rate = self.get_clip_audio(clip_path_left, start_time, left_duration)
                audio_right, _ = self.get_clip_audio(clip_path_right, end_time, right_duration)
                audio  = np.concatenate((audio_left, audio_right), axis=0)
                spectogram = self.get_clip_spectogram(audio, rate)

        # write_video(f'examples/{label}/{self.candidates[idx][0]}.mp4',clip[idxs,:,1:,:],self.network_fps)
        if self.transform and self.visual_stream:
            clip = self.transform(clip)
        

        if self.visual_stream and not self.audio_stream:
            return clip[:,idxs,:,:], label
        
        elif not self.visual_stream and self.audio_stream:
            return spectogram.unsqueeze(0).float(), label
        
        elif self.visual_stream and self.audio_stream:
            return clip[:,idxs,:,:], spectogram.unsqueeze(0).float(), label


    def read_cache_candidates(self):
        dict_cache = json.load(open(self.cache_filename))
        self.candidates = dict_cache['candidates']
        self.labels = dict_cache['labels']
        print('Candidates readed from cache file')

    def set_candidates(self):
        print(f'Setting candidates for {self.mode}')
        self.candidates = []
        self.labels = []
        for idx, row in tqdm.tqdm(self.shots_df.iterrows(),total=len(self.shots_df)):
            this_candidates = []
            this_labels = []
            df = self.shots_df[self.shots_df.video_id==row.video_id]
            # Find positive candidates
            if not (row.shot_left_end - row.shot_left_start > 0.8) and (row.shot_right_end- row.shot_right_start > 0.8):
                continue

            cut_time = (row.shot_left_end + row.shot_right_start)/2
            positive = [row.clip_id, row.shot_left_start, cut_time, row.clip_id, cut_time, row.shot_right_end]
            # Center around zero since annotations come from original scenes
            positive = [x-row.shot_left_start if type(x)!=str else x for x in positive]
            this_candidates.append(positive)
            this_labels.append(1)
            # Find negative candidates
            for i in range(self.negative_positive_ratio):  
                left_found = False
                while not left_found:
                    left = df.sample().iloc[0]
                    if left.shot_left_end - left.shot_left_start > 0.8:
                        cut_time = (left.shot_left_end + left.shot_right_start)/2
                        negative_left = [left.clip_id, left.shot_left_start, cut_time]
                        # Center around zero since annotations come from original scenes
                        negative = [x-left.shot_left_start if type(x)!=str else x for x in negative_left]
                        left_found = True
                right_found = False
                while not right_found:
                    right = df.sample().iloc[0]
                    # sample a different candidate than left
                    # in case of one shot just allow to repeat
                    if right.clip_id == left.clip_id and len(df)>1:
                        continue
                    if right.shot_left_end - right.shot_left_start > 0.8:
                        cut_time = (right.shot_left_end + right.shot_right_start)/2
                        negative_right = [right.clip_id, right.shot_left_start, cut_time]
                        # Center around zero since annotations come from original scenes
                        # create the full negative
                        negative.extend([x-right.shot_left_start if type(x)!=str else x for x in negative_right])
                        right_found = True
                # Center around zero since annotations come from original scenes
                this_candidates.append(negative)
                this_labels.append(0)
            self.candidates.extend(this_candidates)
            self.labels.extend(this_labels)

        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        print(f'Saving cache candidates file in: {self.cache_filename}')
        cache = {'candidates':self.candidates, 'labels':self.labels}
        with open(self.cache_filename, 'w') as f:
            json.dump(cache, f)

