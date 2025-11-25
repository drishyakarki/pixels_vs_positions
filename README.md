# Pixels or Positions? Benchmarking Modalities in Group Activity Recognition
[![arXiv](https://img.shields.io/badge/arXiv-2404.05392-red)](https://arxiv.org/abs/2511.12606)

This repository contains code for our paper:

**Pixels or Positions? Benchmarking Modalities in Group Activity Recognition.** <br>
*Drishya Karki, Merey Ramazanova, Anthony Cioppa, Silvio Giancola, Bernard Ghanem*

## Overview

This repository provides all the codes necessary to reproduce the results on our paper. SoccerNet-GAR dataset used for this paper will be provided through OpenSportsLab HuggingFace repository. We will also provide pretrained models.

## Environment

You can follow either of these options to setup the environment.

### Option 1. Use the `setup.sh` provided.
```bash
chmod +x setup.sh
bash setup.sh
```

### Option 2. Manual installation
```bash
conda create -y -n pvp python=3.10
python -m pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118
python -m pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu118.html
python -m pip install torch-geometric seaborn numpy tqdm transformers opencv-python matplotlib datetime scikit-learn pyarrow==17.0 fastparquet easydict timm
```

## Data
The data SoccerNet-GAR will be provided here.

The dataset is organized in this format
```
data/
├── video_dataset/
│   ├── train/
│   │   ├── train.json
│   │   └── videos/
│   ├── valid/
│   │   ├── valid.json
│   │   └── videos/
│   └── test/
│       ├── test.json
│       └── videos/
└── tracking_dataset/
    ├── train/
    │   ├── train.json
    │   └── videos/
    ├── valid/
    │   ├── valid.json
    │   └── videos/
    └── test/
        ├── test.json
        └── videos/
```

## Execution

### Tracking

```bash
python scripts/train_tracking.py \
    --data-dir data/tracking_dataset \
    --output-dir outputs/tracking \
    --conv-type gin \
    --edge positional \
    --temporal-model attention 
```

#### Graph Convolution Operators `--conv-type`
| Operator | Argument |
|----------|----------|
| Graph Convolutional Network | `graphconv` |
| Graph Attention Network (GATv2) | `gat` |
| GraphSAGE | `sage` |
| Graph Isomorphism Network | `gin` |
| Edge Convolution | `edgeconv` |
| Generalized Aggregation | `gen` |

#### Edge Connectivity `--edge`

| Edge Type | Argument |
|-----------|----------|
| Positional (role-based) | `positional` |
| K-Nearest Neighbors | `knn` |
| Ball K-Nearest Neighbors | `ball_knn` |
| Distance Threshold | `distance` |
| Ball Distance Threshold | `ball_distance` |
| Fully Connected | `full` |
| No Edges | `none` |

#### Temporal Aggregation `--temporal-model`

| Method | Argument |
|--------|----------|
| Mean Pooling | `pool` |
| Max Pooling | `maxpool` |
| Bidirectional LSTM | `bilstm` |
| Temporal Convolutional Network | `tcn` |
| Multi-Head Self-Attention | `attention` |

### Video

**Frozen-backbone**
```bash
python scripts/train_video.py \
    --data-dir data/video_dataset \
    --output-dir outputs/video \
    --backbone videomae2 \
    --temporal-model maxpool \
    --freeze-backbone \
```

**Full-finetuning**
```bash
python scripts/train_video.py \
    --data-dir data/video_dataset \
    --output-dir outputs/video \
    --backbone videomae2 \
    --temporal-model maxpool 
```

#### Backbone Architectures

| Model | Argument | Type |
|-------|----------|------|
| DINOv3 ViT-B/16 | `dinov3` | Image |
| CLIP ViT-B/16 | `clip` | Image |
| VideoMAE | `videomae` | Video |
| VideoMAEv2 | `videomae2` | Video |

*The image backbones have same temporal models as tracking.*

### Evaluation

The best tracking model is provided in the weights folder `weights/tracking/best_model.pt`.
```bash
python scripts/infer_tracking.py \
    --checkpoint weights/tracking/best_model.pt \
    --data-dir data/tracking_dataset \
    --conv-type gin \
    --edge positional \
    --temporal-model attention
```

For video, use the following command.
```bash
python scripts/infer_video.py \
    --checkpoint weights/video/best_model.pt \
    --preprocessed-dir data/video_dataset/preprocessed \
    --backbone videomae2 \
    --temporal-model maxpool \
```

## Trained Models
Trained models will also be provided.

## Contact

If you have any questions related to the code, feel free to contact karkidrishya1@gmail.com.

## References

If you find our work useful, please consider citing our paper.
```
@article{karki2025pixels,
  title={Pixels or Positions? Benchmarking Modalities in Group Activity Recognition},
  author={Karki, Drishya and Ramazanova, Merey and Cioppa, Anthony and Giancola, Silvio and Ghanem, Bernard},
  journal={arXiv preprint arXiv:2511.12606},
  year={2025}
}
```
