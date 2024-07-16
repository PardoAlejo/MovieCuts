# LTC-e2e
MovieCuts and Learning to cut end-to-end pretrained modules

## Requirements

pytorch_lighning 1.1.6
pytorch 1.8

# Installation

Install environmnet:
```bash
conda env create -f environment.yml
```

# Data

**PLEASE READ!**

If you are interested in the data fill this [Google form](https://forms.gle/FUNnZ8wpYRCspTJq9), and in a couple of days I'll send directly to you the  links to download the data. 

## Data Download

After receiving the instruction via email, you can download the data from the web interface. The other option is to use the provided script `moviecuts_downloader.py`, you can use it as follows:

```bash
python data/moviecuts_downloader.py --moviecuts_path {PATH_TO_DOWNLOAD_MOVIECUTS} --download_link {LINK} --password {PASSWORD}
```

LINK and PASSWORD are the same provided in the email's instructions.

The script has several options:
- `--zip_file_index`: To download a specific zip file, if not provided the script will download one by one.
- `--download_and_unzip`: If provided the script will download and unzip the data afterwards.
- `--unzip`: If provided the script will attempt to unzip the data. It will check that all zip files are in the folder, if not, it will throw an error.

**Dependencies for moviecuts_downloader.py**: `pip install google-measurement-protocol` and `pip install tqdm`.

## Videos and Annotations:

*VIDEOS*: To request access to the videos, please fill up [this form](https://forms.gle/FUNnZ8wpYRCspTJq9), agree with all the terms and you will receive and email with a link to access the data.

After receiving the link, please download each one of the zip files (the zip file is partionioed acrross 10 zip files). You can also use the script above to do this process by passing the option --unzip.

After all the files are downloaded (the have to be 12 of them), run the following to combine the files into a single zip:

` zip -s 0 moviecuts.zip --out moviecuts_single_file.zip `

Then you can simply unzip the folder and place it under data:

`unzip moviecuts_single.zip -d ./data/ `

*PRE-TRAINED MODELS/Checkpoints*: Download the pre-trained models and required checkpoints from [here](https://drive.google.com/drive/folders/1SrtYl2E1ftv6tikwiSz_38JjgTplLT-c?usp=sharing).


The folder structure should be as follows:
```
README.md
ltc-env.yml
│
├── data
│   ├── annotated_clips_train.csv
│   ├── annotated_clips_val.csv
│   ├── cut-type-test.json
│   ├── cut-type-train.json
│   ├── cut-type-val.json
│   ├── framed_clips/
│   └── zipped_frames.zip
│
├── checkpoints
|    ├── vggsound_avgpool.pth.tar
|    ├── r2plus1d_18-91a641e6.pth
│    └── epoch=7_Validation_loss=1.91.ckpt
│
├── scripts
├── utils
├── cfg
└── src
```

# Inference


Copy paste the following commands in the terminal. </br>


Load environment: 
```bash
conda activate ltc
cd scripts/
```

Inference on val set 
```bash
sh scripts/run_testing.sh
```

## Expected results:

| **Class**         | **AP** (%)|
|-------------------|-------|
| Cutting on Action | 65.67 |
| Cut Away          | 62.98 |
| Cross Cut         | 34.31 |
| Emphasis Cut      | 31.52 |
| Match Cut         | 2.43  |
| Smash Cut         | 25.01 |
| Reaction Cut      | 83.13 |
| L Cut             | 44.86 |
| J Cut             | 52.02 |
| Speaker-Change Cut| 77.21 |
| Speaker-Change Cut| 77.21 |
| **Mean**          | **47.91** |
</br>

# Training

Copy paste the following commands in the terminal. </br>


Load environment: 
```bash
conda activate ltc
cd scripts/
```

Training on train set and validate on val set 
```bash
sh scripts/run_default_av.sh
```


# If you find this work useful for you research, please cite us:
```bibtex
@inproceedings{pardo2022moviecuts,
  title={Moviecuts: A new dataset and benchmark for cut type recognition},
  author={Pardo, Alejandro and Heilbron, Fabian Caba and Alc{\'a}zar, Juan Le{\'o}n and Thabet, Ali and Ghanem, Bernard},
  booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part VII},
  pages={668--685},
  year={2022},
  organization={Springer}
}
```
