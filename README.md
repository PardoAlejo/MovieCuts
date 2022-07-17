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

The folder structure should be as follows:
```
README.md
ltc-env.yml
│
├── data
│   ├── annotated_clips_train.csv
│   ├── annotated_clips_valv
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

Download the frames and annotations by following:

```bash
mkdir data
```

FRAMES: 
Download the frames [here](https://drive.google.com/file/d/1F57OLtlRxYUMVAFZNZAQ7jj1GCMB3oG-/view?usp=sharing). Unzip the frames under ```data/```.

ANNOTATIONS
Download the annotations files from [here](https://drive.google.com/drive/folders/1crYrtWDDmiNA9eZTfz1D58GQuCN7Im27?usp=sharing).

CHECKPOINTS
Download the pre-trained models from [here](https://drive.google.com/drive/folders/1SrtYl2E1ftv6tikwiSz_38JjgTplLT-c?usp=sharing).

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
Model mAP: 47.91%
AP for cut-on-action: 65.67%
AP for cut-away: 62.98%
AP for cross-cut: 34.31%
AP for emphasis/deemphasis: 31.52%
AP for match-cut: 2.43%
AP for smash-cut: 25.01%
AP for reaction-in/reaction-out-cut: 83.13%
AP for l-cut: 44.86%
AP for j-cut: 52.02%
AP for speaker-change: 77.21%
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


# Cite us
```
@misc{pardo2021moviecuts,
      title={MovieCuts: A New Dataset and Benchmark for Cut Type Recognition}, 
      author={Alejandro Pardo and Fabian Caba Heilbron and Juan León Alcázar and Ali Thabet and Bernard Ghanem},
      year={2021},
      eprint={2109.05569},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
