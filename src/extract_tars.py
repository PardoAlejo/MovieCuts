import glob
import tarfile
import tqdm
import os
import random

frame_path = '../data/movies/frames/*.tar'
frame_tars = glob.glob(frame_path)
out_path = '../data/framed_clips'
random.shuffle(frame_tars)
for tar_filename in tqdm.tqdm(frame_tars):
    clip_name = os.path.basename(tar_filename).replace('.tar','')
    if os.path.exists(f'{out_path}/{clip_name}'):
        continue
    tf = tarfile.open(tar_filename, 'r')
    tf.extractall(path=out_path)
    tf.close()
    os.remove(tar_filename)
