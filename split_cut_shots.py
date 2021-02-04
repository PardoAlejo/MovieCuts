import ffmpeg
import subprocess
import os
from joblib import Parallel, delayed
import pandas as pd
import uuid

def name_cuts(videos_csv):
    clip_names = []
    out_name = videos_csv.replace('.csv', '_named.csv')
    videos_df = pd.read_csv(videos_csv)
    for i in range(len(videos_df)):
        clip_names.append(uuid.uuid4())
    videos_df['clip_name'] = clip_names
    videos_df.to_csv(out_name, index=False)

def clip_one_video(video_filename, clip_filename, start_time, length):
    
    if os.path.exists(clip_filename):
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
        output = subprocess.check_output(command, shell=True,
                                         stderr=subprocess.STDOUT)
        print(f'clip {clip_filename} processed')
    except subprocess.CalledProcessError as err:
        print(command)
        return False, err.output
    # Check if the video was successfully saved.
    status = os.path.exists(clip_filename)
    return status, None

def wrapper_cut_videos(videos_csv, out_dir, num_cores):
    
    videos_df = pd.read_csv(videos_csv)
    videos_list = videos_df.values.tolist()

    results = Parallel(n_jobs=num_cores)(
                    delayed(clip_one_video)(
                        video_filename=f'../data/movies/youtube/{video_id}/{video_id}.mp4',
                        clip_filename=f'{out_dir}/{out_name}.mp4',
                        start_time=start,
                        length=end-start,
                    ) for video_id, start, _, _, end, out_name in videos_list)
    status = []
    for i in range(num_cores):
        status.append(results[i])

if __name__ == "__main__":
    videos_csvs = ['../data/used_cuts_train_named.csv', '../data/used_cuts_val_named.csv']
    out_dir = '../data/movies/clips'
    for video_csv in videos_csvs:
        wrapper_cut_videos(video_csv,out_dir,num_cores=8)