# MNIST Dataset Export for Modelmaker
### - Laavanaya Dhawan, Adithya Thonse 

## Overview
This repository provides:
1. **The MNIST dataset** in `.zip` format, exported into a folder structure compatible with the Modelmaker flow.  
   - Download here: [mnist_classes.zip](https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/mnist_classes.zip)  
2. **A Python script** (`dataset_mnist_creation.py`) used to convert the original [torchvision.datasets.MNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html) into the required format.

Since the MNIST dataset is **not proprietary** but publicly available through `torchvision.datasets`, this script is included to allow anyone to regenerate the dataset if needed.

---



## Script Usage

### Requirements
- Python 3.8+
- PyTorch
- Torchvision

### Install Dependencies

pip install torch torchvision

## Steps to Run

1. Place your raw MNIST .gz files under `./data/MNIST/raw/`.

---

### Auto-download via torchvision (if files are missing):

```python
from torchvision import datasets

datasets.MNIST(root="./data", train=True,  download=True)
datasets.MNIST(root="./data", train=False, download=True)
```
2. Run the export script.

```python
python mnist_dataset_creation.py
```
3. Verify the dataset output under:


```text
mnist_classes/
└── classes/
    ├── 0/
    │   ├── 000000.png
    │   ├── 000001.png
    │   └── ...
    ├── 1/
    │   ├── 000000.png
    │   └── ...
    ├── ...
    └── 9/
        ├── 000000.png
        └── ... 
```

 