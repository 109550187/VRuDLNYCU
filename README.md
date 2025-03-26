# NYCU Computer Vision 2025 Spring HW1
Student ID: 313561002

Name: 梁巍濤

# Introduction
A Python Notebook which main task is to implement a Image Classification Task for a dataset of 21,024 RGB Images for Training and Validation, and 2,344 RGB Images for Testing. The main backbone of the model is ResNet by Pytorch.

# Environment Setup
The notebook is done in a Google Colab environment, leveraging Google Colab's T4 GPU for model training and Google Drive for dataset loading.
## Python Version
Python 3.10
## Required Dependencies
```bash
!pip install torch torchvision matplotlib seaborn scikit-learn tqdm
```
## Required Libraries 
```bash
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import random
from torchvision.models import ResNet101_Weights, resnet101
from tqdm.notebook import tqdm
import IPython
from IPython.display import display
```
## Installation
1. Clone the repository
2. Install dependencies
3. Run the notebook

# Performance
