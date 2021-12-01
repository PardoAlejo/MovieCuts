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
To download the data, please install [gsutil](https://cloud.google.com/storage/docs/gsutil_install). Once [gsutil](https://cloud.google.com/storage/docs/gsutil_install) is installed download the frames and annotations by following:

```bash
mkdir data
```
FRAMES:
```bash
gsutil -m cp -r "gs://pardogl/moviecuts/zipped_frames.zip" data
```

ANNOTATIONS
```bash
gsutil -m cp \
  "gs://pardogl/moviecuts/annotations/cut-type-test.json" \
  "gs://pardogl/moviecuts/annotations/cut-type-train.json" \
  "gs://pardogl/moviecuts/annotations/cut-type-val.json" \
  data
```
PRETRAINED MODELS:
```bash
gsutil -m cp \
  "gs://pardogl/moviecuts/model_checkpoints/epoch=7_Validation_loss=1.91.ckpt" \
  "gs://pardogl/moviecuts/model_checkpoints/r2plus1d_18-91a641e6.pth" \
  "gs://pardogl/moviecuts/model_checkpoints/vggsound_avgpool.pth.tar" \
  checkpoints
```

The folder structure should be as follows:
```
README.md
ltc-env.yml
│
├── data
│   ├── cut-type-test.json
│   ├── cut-type-train.json
│   ├── cut-type-val.json
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

## Expected results (Table 1 of the Paper):

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
