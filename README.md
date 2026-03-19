Physics-Coupled Frequency Dynamic Adaptation Network for Domain Generalized Underwater Object Detection

This repository contains the official PyTorch implementation for our paper accepted by **ACM Multimedia (ACM MM) 2025**:
 - [Physics-Coupled Frequency Dynamic Adaptation Network for Domain Generalized Underwater Object Detection](https://dl.acm.org/doi/10.1145/3746027.3755829)

# 📂 Installation

 - Clone this repository and enter it:
   ```Shell
   git clone https://github.com/White-cat-ed/HyUOD
   cd HyUOD
   ```
 - Set up the environment using Conda:
   ```Shell
   # Create a conda environment with Python 3.8
   conda create -n hyuod python=3.8 -y
   conda activate hyuod
   ```
 - Install PyTorch (2.1.1+cu118) and TorchVision:
   ```Shell
   pip install torch==2.1.1+cu118 torchvision==0.16.1+cu118 torchaudio==2.1.1+cu118 --index-url https://download.pytorch.org/whl/cu118
   ```
 - Install the remaining packages from `requirements.txt`:
   ```Shell
   pip install -r requirements.txt
   ```

# 📦 Data Preparation

First, download your target underwater object detection dataset and organize it into the standard YOLO dataset format. The initial directory structure should look like this:

```text
dataset_root/
├── images/
│   ├── train/
│   └── test/
└── labels/
    ├── train/
    └── test/
```

To accelerate the training process, we pre-generate the transmission map (`t`) and global atmospheric light (`A`). Run the `ta_generate.py` script to generate these physical parameters:

```Shell
python ta_generate.py /path/to/dataset_root/images /path/to/dataset_root
```

After generation, your dataset directory structure should be updated to:

```text
dataset_root/
├── images/
│   ├── train/
│   └── test/
├── labels/
│   ├── train/
│   └── test/
├── t/
│   ├── train/
│   └── test/
└── a/
    ├── train/
    └── test/
```

Finally, create your dataset YAML file. Here is an example using the **DUO dataset**:

```yaml
path: /path/to/dataset_root  # dataset root dir
train: images/train 
train_t: t/train 
train_a: a/train 
val: images/test 
nc: 4
names:
  0: holothurian
  1: echinus
  2: scallop
  3: starfish
```

# Usage

## 🚀 Training
To train the model, run the following command using the provided configuration and your dataset yaml:
```Shell
python train.py train_yaml/hyuod.yaml /path/to/your/data_yaml.yaml
```

## 🧪 Evaluation

We provide pre-trained weights for several underwater datasets (including **DUO**, **RUOD**, **SUODAC2020**, and **URPC2020**), which are located in the `weights/` directory.

To evaluate a trained model, run the following command with your saved weights:
```Shell
# Example: Evaluating on the DUO dataset using the provided pre-trained weights
python val.py weights/DUO.pt /path/to/your/data_yaml.yaml
```

# 📜 Citation

If you use HyUOD or this code base in your work, please cite our paper:

```bibtex
@inproceedings{10.1145/3746027.3755829,
author = {Luo, Linxuan and Mu, Pan and Bai, Cong},
title = {Physics-Coupled Frequency Dynamic Adaptation Network for Domain Generalized Underwater Object Detection},
year = {2025},
isbn = {9798400720352},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {[https://doi.org/10.1145/3746027.3755829](https://doi.org/10.1145/3746027.3755829)},
doi = {10.1145/3746027.3755829},
booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
pages = {2284–2293},
numpages = {10},
keywords = {deep learning, domain generalized underwater object detection, hypernetwork, underwater object detection},
location = {Dublin, Ireland},
series = {MM '25}
}
```
```
