# Pixels or Positions? Benchmarking Modalities in Group Activity Recognition

This repository contains the code and configuration files needed to reproduce the experiments from our paper. We benchmark video-based and tracking-based approaches for Group Activity Recognition (GAR) on the SoccerNet-GAR dataset.

[[Paper]](https://arxiv.org/abs/2511.12606) [[Dataset]](https://huggingface.co/datasets/OpenSportsLab/soccernetpro-classification-GAR)

## Key Results

Our tracking baseline (GIN + MaxPool + Positional Edges, **180K** parameters) achieves **77.8%** balanced accuracy and **57.0%** macro F1, outperforming the best video model (VideoMAEv2-B finetuned, 86.3M parameters) by **16.9 pp** in balanced accuracy and **6.9 pp** in macro F1, while training **7x** faster with **479x** fewer parameters.

| Modality | Model | Params | Bal. Acc. | F1 | Training |
|---|---|---|---|---|---|
| **Tracking** | GIN + MaxPool + Positional | **180K** | **77.8** | **57.0** | 4 GPU.h |
| Video | VideoMAEv2-B (finetuned) | 86.3M | 60.9 | 50.1 | 28 GPU.h |

## Reproducing Tracking Results

Follow the six steps below in order. Each step tells you exactly which directory you should be in before running the commands. Do not skip ahead and do not change any paths. If you follow these steps as-is, you will reproduce the tracking baseline end-to-end.


### Step 1: Create your workspace and clone this repository

Open a terminal. Starting from your home directory (or anywhere you like to keep projects), run:

```bash
mkdir reproducing-sn-gar
cd reproducing-sn-gar
git clone https://github.com/drishyakarki/pixels_vs_positions.git
```

Confirm you are inside the workspace folder:

```bash
pwd
# should end in /reproducing-sn-gar
```

Your directory layout at this point:

```
reproducing-sn-gar/          <-- you are here
└── pixels_vs_positions/
```

### Step 2: Create and activate the conda environment

Stay inside `reproducing-sn-gar/`. Run these commands one at a time:

```bash
conda create -n osl python=3.12 pip -y
conda activate osl
pip install opensportslib==0.1.2
opensportslib setup
opensportslib setup --pyg
```

After this step your terminal prompt should start with `(osl)`. That means the environment is active. Every command from here on assumes `(osl)` is active.

### Step 3: Download the tracking dataset

You should still be inside `reproducing-sn-gar/`. Create a file named `download_data.py` with the following contents:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="OpenSportsLab/soccernetpro-classification-GAR",
    repo_type="dataset",
    revision="tracking",
    local_dir="sngar-tracking",
)
```

Then run it:

```bash
python download_data.py
```

When the download finishes, your directory layout will be:

```
reproducing-sn-gar/          <-- you are here
├── pixels_vs_positions/
├── download_data.py
└── sngar-tracking/
    ├── train.zip
    ├── valid.zip
    ├── test.zip
    └── annotations_*.json
```

### Step 4: Unzip the dataset splits

From `reproducing-sn-gar/`, enter the dataset folder, unzip each split, delete the zip files, then return to the workspace:

```bash
cd sngar-tracking
unzip train.zip && rm train.zip
unzip valid.zip && rm valid.zip
unzip test.zip && rm test.zip
cd ..
```

Confirm you are back in the workspace folder:

```bash
pwd
# should end in /reproducing-sn-gar
```

### Step 5: Create the training script

You must create this script inside `reproducing-sn-gar/`, NOT inside the `pixels_vs_positions/` folder. Create a file named `run_tracking.py` with exactly the following contents:

```python
from opensportslib import model

if __name__ == '__main__':
    myModel = model.classification(
        # path to the config file from the pixels_vs_positions repository.
        # this one reproduces the baseline tracking experiment from Table 2.
        config="pixels_vs_positions/main_tracking_gin_positional_maxpool.yaml",
        # path to the dataset directory you downloaded in Step 3.
        data_dir="sngar-tracking",
    )

    myModel.train(
        train_set="sngar-tracking/annotations_train.json",
        valid_set="sngar-tracking/annotations_valid.json",
        use_ddp=False,
    )

    myModel.infer(test_set="sngar-tracking/annotations_test.json")
```

Your final directory layout should look like this:

```
reproducing-sn-gar/          <-- you are here
├── pixels_vs_positions/
├── sngar-tracking/
├── download_data.py
└── run_tracking.py
```

### Step 6: Train and evaluate

From `reproducing-sn-gar/`, simply run:

```bash
python run_tracking.py
```

That is it. The script will train the model, run validation, and then report the final metrics on the test set. You do not need to edit any config files. You do not need to change any paths. Every path in the script is relative to the workspace folder.

### Running a different experiment

To reproduce a different row from the paper, only change the `config=` line in `run_tracking.py`. For example, to run the EdgeConv graph layer ablation from Table 4:

```python
config="pixels_vs_positions/tracking-configs/backbones_table_4/edgeconv.yaml",
```

The full list of config files is in the next section.

## Configuration Files

Each config file corresponds to one row in the paper's tables. The directory structure mirrors the ablation tables:

```
.
├── main_tracking_gin_positional_maxpool.yaml      # Table 2: Tracking baseline
├── main_video_videomae2_full_finetuned.yaml       # Table 2: Video baseline
│
├── tracking-configs/
│   ├── backbones_table_4/                         # Table 4: Graph Layer Ablation
│   │   ├── gin.yaml                               #   GIN (baseline)
│   │   ├── graphconv.yaml                         #   GraphConv
│   │   ├── edgeconv.yaml                          #   EdgeConv
│   │   ├── gatv2.yaml                             #   GATv2
│   │   ├── gen.yaml                                #   GEN
│   │   └── graphsage.yaml                         #   GraphSAGE
│   │
│   ├── edges_table_5/                             # Table 5: Edge Connectivity Ablation
│   │   ├── positional.yaml                        #   Positional (baseline)
│   │   ├── fully_connected.yaml                   #   Fully Connected
│   │   ├── distance.yaml                          #   Distance (r=15m)
│   │   ├── knn.yaml                               #   KNN (k=8)
│   │   ├── ball_distance.yaml                     #   Ball Distance (r=20m)
│   │   ├── ball_knn.yaml                          #   Ball KNN (k=8)
│   │   └── no_edges.yaml                          #   No Edges
│   │
│   └── temporal_neck_table_6/                     # Table 6: Temporal Aggregation Ablation
│       ├── maxpool.yaml                           #   MaxPool (baseline)
│       ├── avgpool.yaml                           #   AvgPool
│       ├── attention.yaml                         #   Attention
│       ├── tcn.yaml                                #   TCN
│       └── bilstm.yaml                            #   BiLSTM
│
└── video-configs/
    ├── frozen_table_3/                            # Table 3: Video Backbone (frozen)
    │   ├── videomae.yaml                          #   VideoMAE-B frozen
    │   └── videomae2.yaml                         #   VideoMAEv2-B frozen
    │
    └── fully_finetune_table_3/                    # Table 3: Video Backbone (finetuned)
        ├── videomae.yaml                          #   VideoMAE-B finetuned
        └── videomae2.yaml                         #   VideoMAEv2-B finetuned
```

## Reproducing Video Results

The video reproduction follows the same six-step structure as the tracking reproduction above. Steps 1 and 2 are identical (create the workspace, clone the repo, set up the `osl` conda environment with `opensportslib==0.1.2`). Only the dataset download, the unzip step, and the training script differ.

### Step 3 (video): Download the video dataset

From `reproducing-sn-gar/`, create a file named `download_video_data.py`:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="OpenSportsLab/soccernetpro-classification-GAR",
    repo_type="dataset",
    revision="frames",
    local_dir="sngar-frames",
)
```

Run it:

```bash
python download_video_data.py
```

### Step 4 (video): Concatenate and unzip the splits

The video training split is stored as a multi-part zip, so it must be concatenated before unzipping. From `reproducing-sn-gar/`:

```bash
cd sngar-frames
cat train.zip.part_aa train.zip.part_ab > train.zip
unzip train.zip && unzip valid.zip && unzip test.zip
rm train.zip.part_aa train.zip.part_ab train.zip valid.zip test.zip
cd ..
```

### Step 5 (video): Create the training script

From `reproducing-sn-gar/` (not inside `pixels_vs_positions/`), create a file named `run_video.py`:

```python
from opensportslib import model

if __name__ == '__main__':
    myModel = model.classification(
        # path to the video baseline config from the pixels_vs_positions repository.
        # this one reproduces the VideoMAEv2-B finetuned result from Table 2.
        config="pixels_vs_positions/main_video_videomae2_full_finetuned.yaml",
        # path to the video dataset directory you downloaded above.
        data_dir="sngar-frames",
    )

    myModel.train(
        train_set="sngar-frames/annotations_train.json",
        valid_set="sngar-frames/annotations_valid.json",
        use_ddp=False,
    )

    myModel.infer(test_set="sngar-frames/annotations_test.json")
```

### Step 6 (video): Train and evaluate

From `reproducing-sn-gar/`:

```bash
python run_video.py
```

To reproduce a different video row from Table 3, change the `config=` line to one of the files under `pixels_vs_positions/video-configs/`.

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