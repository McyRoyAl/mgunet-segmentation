"""
Default configuration for the MGUnet project.
"""

# Model Architecture
MODEL_CONFIG = {
    'in_channels': 4,  # 4 for neuron segmentation, 3 for carvana
    'out_channels': 2,
    'depth': 5,
    'width': 16,
    'hyper_depth': 4,
    'mid_out_channels': 16,
}

# Training Configuration
TRAIN_CONFIG = {
    'epochs': 100,
    'batch_size': 1,
    'learning_rate': 1e-3,
    'save_freq': 0,
    'save_model': 0,
}

# Data Augmentation
AUGMENTATION_CONFIG = {
    'train_flip_prob': 0.5,
    'val_flip_prob': 0.0,
    'crop_size': None,  # Set to tuple (height, width) if needed
}

# Device Configuration
DEVICE_CONFIG = {
    'device': 'cpu',  # Change to 'cuda' if GPU is available
}

# Paths
PATH_CONFIG = {
    'checkpoint_path': 'checkpoints',
    'model_name': 'model_test',
} 