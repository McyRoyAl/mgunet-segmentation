# MGUnet: Multigrid CNN for Image Segmentation

## Abstract

Semantic image segmentation is the process of labeling each pixel of an image with its corresponding class. UNet is a popular Convolutional Neural Network architecture, originally proposed for biomedical image segmentation, but now widely used in a range of additional segmentation tasks. Many architectures have been built on top of the UNet framework, such as U-Net++, UNet2, DoubleUnet, MultiResUNet, and many others.

We propose a new architecture called MGUnet, which is a multigrid CNN consisting of n concatenated interconnected UNet components. MGUnet can be used for classification and weakly supervised segmentation tasks as well. We evaluate MGUnet on medical segmentation datasets covering various medical techniques such as dermatoscopy, colonoscopy, and microscopy, and compare it with the UNet and DoubleUnet architectures. Our results show that MGUnet outperforms the UNet model on all datasets except for one. In addition, we compared the model with the DoubleUnet architecture which has been shown to outperform several other improved UNet architectures, such as Multi-ResUNet, Unet++ in terms of dice coefficient and IOU loss.

## Background

Image segmentation has a long history and has traditionally been performed manually by experts with prior knowledge and experience in the field. The manual approach involved the use of techniques such as thresholding, edge detection, and region growing to identify and isolate different regions or objects within an image. With advancements in computer vision and machine learning, automated approaches for image segmentation have become more prevalent and are now used widely in various applications. These automated approaches rely on sophisticated algorithms to perform the segmentation, and are typically more efficient and less prone to human error compared to the manual approach. In recent years, the common approach for solving image segmentation problems has shifted towards treating the task as a pixel-wise classification problem. This is accomplished by obtaining the classification information of each pixel through the use of Convolutional Neural Networks (CNNs). This approach has become prevalent due to the improved accuracy and efficiency achieved through the use of deep learning techniques.

## Features

- Image preprocessing with automatic mask generation
- UNet model implementation with skip connections
- Training with validation and checkpointing
- Progress tracking and logging
- GPU support

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate conda environment:
```bash
# Create environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate mgunet
```

3. Verify installation:
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

## Data Preparation

### Dataset Sources

The datasets used in this project can be found in the final paper. These include medical imaging datasets from various sources.

### Data Organization

1. Create the required directories:
```bash
mkdir -p data/raw data/processed
```

2. Download the datasets from the sources mentioned in the final paper
3. Place your downloaded images in the `data/raw` directory

### Data Preprocessing

1. Place your raw images in the `data/raw` directory
2. Run the preprocessing script:
```bash
python scripts/preprocess.py --input_dir data/raw --output_dir data/processed/train --size 256
```

The preprocessing script will:
- Resize images to 256x256 pixels
- Generate corresponding masks
- Save processed images and masks in the appropriate directories

## Training

### Basic Training

1. Create the checkpoints directory:
```bash
mkdir -p checkpoints
```

2. Train the model with default parameters:
```bash
python scripts/train.py
```

### Advanced Training

For more control over the training process, you can specify various parameters:
```bash
python scripts/train.py \
    --data_dir data/processed/train \
    --batch_size 4 \
    --epochs 50 \
    --lr 1e-4 \
    --val_split 0.2 \
    --checkpoint_dir checkpoints
```

Training options:
- `--data_dir`: Directory containing processed images and masks
- `--batch_size`: Number of images per batch (default: 4)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 1e-4)
- `--val_split`: Validation split ratio (default: 0.2)
- `--checkpoint_dir`: Directory to save model checkpoints

## Project Structure

```
.
├── data/               # Data directory
│   ├── raw/           # Raw images
│   └── processed/     # Processed images and masks
├── checkpoints/       # Saved model checkpoints
├── scripts/           # Python scripts
│   ├── preprocess.py  # Image preprocessing
│   └── train.py      # Model training
└── environment.yml    # Conda environment configuration
```

## Model Architecture

The project implements a UNet architecture with:
- Encoder path with 4 levels
- Bottleneck layer
- Decoder path with skip connections
- Batch normalization and ReLU activation
- Final 1x1 convolution for segmentation

## Training Process

The training process includes:
- Automatic train/validation split
- Learning rate scheduling
- Model checkpointing
- Progress tracking
- GPU support when available

## Requirements

- Python 3.7+
- PyTorch 2.0+
- torchvision
- numpy
- Pillow
- tqdm

