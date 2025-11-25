#!/bin/bash
conda create -y -n pvp python=3.10
conda activate pvp

# pytorch with CUDA 11.8
pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# torch-geometric
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu118.html
pip install torch-geometric

pip install -r requirements.txt