import ffmpeg
import subprocess
import os
from joblib import Parallel, delayed
import pandas as pd
import uuid
import random
import tqdm

def name_cuts(videos_csv):
    clip_names = []
    out_name = videos_csv.replace('.csv', '_named.csv')
    videos_df = pd.read_csv(videos_csv)
    for i in range(len(videos_df)):
        clip_names.append(uuid.uuid4())
    videos_df['clip_name'] = clip_names
    videos_df.to_csv(out_name, index=False)

def filter(target_csv_file,
            clip_info, 
            movie_info, 
            durations, 
            clips_path, 
            genre_to_filter='Animation'):

    original_df = pd.read_csv(target_csv_file)
    new_df = original_df.copy()
    scene_info = pd.read_csv(clip_info)
    movie_info = pd.read_csv(movie_info)
    durations = pd.read_csv(durations)
    
    df_per_scene = original_df.groupby('video_id')
    
    filtered_animated_clips = []
    filtered_short_clips = []
    num_without_genre_or_duration = 0
    
    for video_id, this_df in tqdm.tqdm(df_per_scene):
        imdb_id = scene_info.loc[scene_info['videoid'] == video_id, 'imdbid'].values[0]
        try:
            movie_genre = movie_info.loc[movie_info['imdbid'] == imdb_id, 'genre'].values[0]
            duration = durations.loc[durations['videoid'] == video_id, 'duration'].values[0]
        except:
            num_without_genre_or_duration += 1
            continue
        
        if genre_to_filter in movie_genre:
            for idx, row in this_df.iterrows():
                clip_name = row['clip_name']
                filtered_animated_clips.append(clip_name)
                new_df = new_df.drop(index=idx)
    
    new_df.to_csv(target_csv_file.replace('.csv','_no_animations.csv'), index=False)

def pick_clips(target_csv_train_file, target_csv_val_file, number_to_keep):

    original_train_df = pd.read_csv(target_csv_train_file)
    original_val_df = pd.read_csv(target_csv_val_file)
    
    total_samples = len(original_train_df) + len(original_val_df)

    ratio_to_keep = number_to_keep/total_samples
    num_samples_train = int(len(original_train_df)*ratio_to_keep)
    num_samples_val = int(len(original_val_df)*ratio_to_keep)

    new_train_df = original_train_df.sample(n=num_samples_train)
    new_val_df = original_val_df.sample(n=num_samples_val)

    new_train_df.to_csv('../data/train_samples_to_annotate.csv', index=False)
    new_val_df.to_csv('../data/val_samples_to_annotate.csv', index=False)

    print(f'total kept videos train:{len(new_train_df)}')
    print(f'total kept videos val:{len(new_val_df)}')
    print(f'total videos: {len(new_train_df)+len(new_val_df)}')

def clip_one_video(video_filename, clip_filename, start_time, length):
    
    base_name = os.path.basename(clip_filename)
    print(base_name)

    if os.path.exists(clip_filename):
        if os.stat(clip_filename).st_size > 50000:
            return 'Already processed', None

    command = ['ffmpeg',
               '-i', '"%s"' % video_filename,
               '-ss', str(start_time),
               '-t', str(length),
               '-c:v', 'libx264',
               '-c:a', 'copy',
               '-crf', '19',
               '-threads', '0',
               '"%s"' % clip_filename]
    command = ' '.join(command)
    try:
        size = 0
        count = 0
        while size < 50000:
            if count > 10:
                break
            output = subprocess.check_output(command, shell=True,
                                            stderr=subprocess.STDOUT)
            size = os.stat(clip_filename).st_size
            count+=1
        if size<50000:
            print(f'clip {clip_filename} corrupted')

    except subprocess.CalledProcessError as err:
        print(command)
        return False, err.output
    # Check if the video was successfully saved.
    status = os.path.exists(clip_filename)
    return status, None

def extract_clips(target_csv_file, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    videos_df = pd.read_csv(target_csv_file)
    videos_list = videos_df.values.tolist()
    random.shuffle(videos_list)

    for elmnt in videos_list:
        video_id = elmnt[0]
        clip_name = elmnt[-1]
        video_filename = f'../data/movies/youtube/{video_id}/{video_id}.mp4'
        clip_filename=f'{out_dir}/{clip_name}.mp4'
        start_time = elmnt[1]
        length = elmnt[4] - start_time
        results = clip_one_video(video_filename, clip_filename, start_time, length)

def wrapper_cut_videos(videos_csv, out_dir, existent_list, num_cores):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    videos_df = pd.read_csv(videos_csv)
    videos_list = videos_df.values.tolist()
    random.shuffle(videos_list)

    results = Parallel(n_jobs=num_cores)(
                    delayed(clip_one_video)(
                        video_filename=f'../data/movies/youtube/{video_id}/{video_id}.mp4',
                        clip_filename=f'{out_dir}/{out_name}.mp4',
                        start_time=start,
                        length=end-start,
                        existent_list=existent_list
                    ) for video_id, start, _, _, end, out_name in videos_list)
    status = []
    for i in range(num_cores):
        status.append(results[i])

if __name__ == "__main__":
    videos_csvs = ['../data/used_cuts_train_named.csv', '../data/used_cuts_val_named.csv']
    clip_info = '../data/movies/metadata/clips.csv'
    movie_info = '../data/movies/metadata/movie_info.csv'
    durations = '../data/movies/metadata/durations.csv'
    clips_path = '../data/movies/clips_4.0'

    target_csv_train_file = '../data/used_cuts_train_named_no_animations.csv' 
    target_csv_val_file = '../data/used_cuts_val_named_no_animations.csv'
    
    for csv in videos_csvs:
        if os.path.exists(target_csv_train_file) and 'train' in csv:
            continue

        if os.path.exists(target_csv_val_file) and 'val' in csv:
            continue

        filter(csv,
            clip_info, 
            movie_info, 
            durations, 
            clips_path)
    
    if not os.path.exists('../data/train_samples_to_annotate.csv'):
        pick_clips(target_csv_train_file, target_csv_val_file, number_to_keep=200000)

    picked_csvs = [target_csv_train_file, target_csv_val_file]
    random.shuffle(picked_csvs)
    out_dir = '../data/movies/clips'
    extract_clips(picked_csvs[0], out_dir)
    