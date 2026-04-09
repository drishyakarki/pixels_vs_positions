# Pixels or Positions? Benchmarking Modalities in Group Activity Recognition

This repository contains the code and configuration files needed to reproduce the experiments from our paper. We benchmark video-based and tracking-based approaches for Group Activity Recognition (GAR) on the SoccerNet-GAR dataset.

[[Paper]](https://arxiv.org/abs/2511.12606) [[Dataset]](https://huggingface.co/datasets/OpenSportsLab/soccernetpro-classification-GAR)

## Key Results

Our tracking baseline (GIN + MaxPool + Positional Edges, **180K** parameters) achieves **77.8%** balanced accuracy and **57.0%** macro F1, outperforming the best video model (VideoMAEv2-B finetuned, 86.3M parameters) by **16.9 pp** in balanced accuracy and **6.9 pp** in macro F1, while training **7x** faster with **479x** fewer parameters.

| Modality | Model | Params | Bal. Acc. | F1 | Training |
|---|---|---|---|---|---|
| **Tracking** | GIN + MaxPool + Positional | **180K** | **77.8** | **57.0** | 4 GPU.h |
| Video | VideoMAEv2-B (finetuned) | 86.3M | 60.9 | 50.1 | 28 GPU.h |

## Installation

This project uses [OpenSportsLib](https://github.com/OpenSportsLab/opensportslib) for training and evaluation.

```bash
pip install opensportslib
```

Or clone and install from source:
```bash
git clone https://github.com/OpenSportsLab/opensportslib.git
cd opensportslib
pip install -e .
```

## Dataset

Download the SoccerNet-GAR dataset from [HuggingFace](https://huggingface.co/datasets/OpenSportsLab/soccernetpro-classification-GAR).

**Tracking data:**

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="OpenSportsLab/soccernetpro-classification-GAR",
    repo_type="dataset",
    revision="tracking-parquet",
    local_dir="sngar-tracking",
)
```

**Video data:**

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="OpenSportsLab/soccernetpro-classification-GAR",
    repo_type="dataset",
    revision="frames",
    local_dir="sngar-frames",
)
```

The video training split is stored as a multi-part zip:

```bash
cd sngar-frames
cat train.zip.part_aa train.zip.part_ab > train.zip
unzip train.zip && unzip valid.zip && unzip test.zip
rm train.zip.part_aa train.zip.part_ab train.zip valid.zip test.zip
```

## Usage

All experiments follow the same pattern. Point the config to your data directory, then train and evaluate:

```python
from opensportslib import model

if __name__ == '__main__':
    myModel = model.classification(
        config="path/to/config.yaml",
        data_dir="path/to/data",
    )

    myModel.train(
        train_set="path/to/data/annotations_train.json",
        valid_set="path/to/data/annotations_valid.json",
        use_ddp=False
    )

    myModel.infer(test_set="path/to/data/annotations_test.json")
```

> **Note:** Update `DATA.data_dir` in the YAML config to match your local data path before running.

## Configuration Files

Each config file corresponds to one row in the paper's tables. The directory structure mirrors the ablation tables:

```
.
├── main_tracking_gin_positional_maxpool.yaml      # Table 2: Tracking baseline
├── main_video_videomae2_full_finetuned.yaml        # Table 2: Video baseline
│
├── tracking-configs/
│   ├── backbones_table_4/                          # Table 4: Graph Layer Ablation
│   │   ├── gin.yaml                                #   GIN (baseline)
│   │   ├── graphconv.yaml                          #   GraphConv
│   │   ├── edgeconv.yaml                           #   EdgeConv
│   │   ├── gatv2.yaml                              #   GATv2
│   │   ├── gen.yaml                                #   GEN
│   │   └── graphsage.yaml                          #   GraphSAGE
│   │
│   ├── edges_table_5/                              # Table 5: Edge Connectivity Ablation
│   │   ├── positional.yaml                         #   Positional (baseline)
│   │   ├── fully_connected.yaml                    #   Fully Connected
│   │   ├── distance.yaml                           #   Distance (r=15m)
│   │   ├── knn.yaml                                #   KNN (k=8)
│   │   ├── ball_distance.yaml                      #   Ball Distance (r=20m)
│   │   ├── ball_knn.yaml                           #   Ball KNN (k=8)
│   │   └── no_edges.yaml                           #   No Edges
│   │
│   └── temporal_neck_table_6/                      # Table 6: Temporal Aggregation Ablation
│       ├── maxpool.yaml                            #   MaxPool (baseline)
│       ├── avgpool.yaml                            #   AvgPool
│       ├── attention.yaml                          #   Attention
│       ├── tcn.yaml                                #   TCN
│       └── bilstm.yaml                             #   BiLSTM
│
└── video-configs/
    ├── frozen_table_3/                             # Table 3: Video Backbone (frozen)
    │   ├── videomae.yaml                           #   VideoMAE-B frozen
    │   └── videomae2.yaml                          #   VideoMAEv2-B frozen
    │
    └── fully_finetune_table_3/                     # Table 3: Video Backbone (finetuned)
        ├── videomae.yaml                           #   VideoMAE-B finetuned
        └── videomae2.yaml                          #   VideoMAEv2-B finetuned
```

### Reproducing a Specific Result

For example, to reproduce the GIN + Attention + Positional result from Table 6:

```python
from opensportslib import model

if __name__ == '__main__':
    myModel = model.classification(
        config="tracking-configs/temporal_neck_table_6/attention.yaml",
        data_dir="sngar-tracking",
    )

    myModel.train(
        train_set="sngar-tracking/annotations_train.json",
        valid_set="sngar-tracking/annotations_valid.json",
    )

    myModel.infer(test_set="sngar-tracking/annotations_test.json")
```

## Full Results

### Graph Layer Ablation (Table 4)

All models use MaxPool temporal aggregation and positional edges. Mean +/- std over 5 seeds.

| Backbone | Params | Bal. Acc. | F1 |
|---|---|---|---|
| **GIN** | **180K** | **77.8 +/- 0.7** | **57.0 +/- 0.9** |
| GraphConv | 174K | 76.3 +/- 1.1 | 56.5 +/- 1.8 |
| GraphSAGE | 421K | 75.9 +/- 1.1 | 56.9 +/- 1.7 |
| GEN | 341K | 72.8 +/- 3.0 | 54.7 +/- 3.1 |
| GATv2 | 177K | 61.8 +/- 5.4 | 42.7 +/- 4.5 |
| EdgeConv | 174K | 55.6 +/- 12.1 | 35.3 +/- 11.9 |

### Edge Connectivity Ablation (Table 5)

All models use GIN backbone and MaxPool. Mean +/- std over 5 seeds.

| Edge Type | Bal. Acc. | F1 |
|---|---|---|
| **Positional** | **77.8 +/- 0.7** | **57.0 +/- 0.9** |
| Fully Connected | 71.4 +/- 2.4 | 49.8 +/- 1.6 |
| No Edges | 68.9 +/- 2.7 | 47.0 +/- 3.1 |
| Ball Distance | 68.6 +/- 1.0 | 45.3 +/- 1.3 |
| Distance | 68.0 +/- 1.1 | 45.4 +/- 1.3 |
| Ball KNN | 67.0 +/- 0.9 | 45.9 +/- 1.0 |
| KNN | 66.7 +/- 1.4 | 46.4 +/- 1.6 |

### Temporal Aggregation Ablation (Table 6)

All models use GIN backbone and positional edges. Mean +/- std over 5 seeds.

| Temporal | Params | Bal. Acc. | F1 |
|---|---|---|---|
| **MaxPool** | **180K** | **77.8 +/- 0.7** | 57.0 +/- 0.9 |
| Attention | 197K | 77.3 +/- 0.9 | 58.3 +/- 1.2 |
| TCN | 205K | 75.5 +/- 0.3 | **58.4 +/- 0.6** |
| BiLSTM | 350K | 75.3 +/- 2.6 | 57.9 +/- 1.0 |
| AvgPool | 180K | 66.6 +/- 0.9 | 42.5 +/- 1.5 |

### Video Backbone Ablation (Table 3)

| Backbone | Frozen | Bal. Acc. | F1 |
|---|---|---|---|
| **VideoMAEv2-B** | No | **60.9** | **50.1** |
| VideoMAE-B | No | 55.2 | 49.4 |
| VideoMAEv2-B | Yes | 49.3 | 30.7 |
| VideoMAE-B | Yes | 34.6 | 20.6 |

## Citation

```bibtex
@article{karki2025pixels,
  title={Pixels or Positions? Benchmarking Modalities in Group Activity Recognition},
  author={Karki, Drishya and Ramazanova, Merey and Cioppa, Anthony and Giancola, Silvio and Ghanem, Bernard},
  journal={arXiv preprint arXiv:2511.12606},
  year={2025}
}
```

## Acknowledgment

This research was supported by King Abdullah University of Science and Technology (KAUST) - Center of Excellence for Generative AI, under award number 5940.