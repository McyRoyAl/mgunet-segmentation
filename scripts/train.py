import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
import logging
import argparse
from tqdm import tqdm
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImageDataset(Dataset):
    """Dataset class for loading and preprocessing image-mask pairs.
    
    This dataset class handles loading of image and mask pairs, applies transformations,
    and provides proper error handling for corrupted files.
    
    Args:
        image_paths (list): List of paths to input images
        mask_paths (list): List of paths to corresponding masks
        transform (callable, optional): Transform to be applied on images and masks
    """
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            mask = Image.open(self.mask_paths[idx]).convert('L')
            
            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)
            
            return image, mask
        except Exception as e:
            logger.error(f"Error loading image {self.image_paths[idx]}: {str(e)}")
            # Return a zero tensor as fallback
            return torch.zeros((3, 256, 256)), torch.zeros((1, 256, 256))

class UNet(nn.Module):
    """UNet model for image segmentation.
    
    This implementation includes:
    - Encoder path with 4 levels
    - Bottleneck layer
    - Decoder path with skip connections
    - Batch normalization and ReLU activation
    - Final 1x1 convolution for segmentation
    
    Args:
        in_channels (int): Number of input channels (default: 3)
        out_channels (int): Number of output channels (default: 1)
        features (int): Number of features in first layer (default: 64)
    """
    def __init__(self, in_channels=3, out_channels=1, features=64):
        super().__init__()
        
        # Encoder
        self.enc1 = self._block(in_channels, features)
        self.enc2 = self._block(features, features*2)
        self.enc3 = self._block(features*2, features*4)
        self.enc4 = self._block(features*4, features*8)
        
        # Bottleneck
        self.bottleneck = self._block(features*8, features*16)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(features*16, features*8, kernel_size=2, stride=2)
        self.dec4 = self._block(features*16, features*8)
        self.upconv3 = nn.ConvTranspose2d(features*8, features*4, kernel_size=2, stride=2)
        self.dec3 = self._block(features*8, features*4)
        self.upconv2 = nn.ConvTranspose2d(features*4, features*2, kernel_size=2, stride=2)
        self.dec2 = self._block(features*4, features*2)
        self.upconv1 = nn.ConvTranspose2d(features*2, features, kernel_size=2, stride=2)
        self.dec1 = self._block(features*2, features)
        
        # Final convolution
        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)
        
    def _block(self, in_channels, out_channels):
        """Create a block of two convolutional layers with batch normalization and ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(nn.MaxPool2d(2)(enc1))
        enc3 = self.enc3(nn.MaxPool2d(2)(enc2))
        enc4 = self.enc4(nn.MaxPool2d(2)(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(nn.MaxPool2d(2)(enc4))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        return self.final_conv(dec1)

def load_data(data_dir):
    """Load image and mask paths from the data directory.
    
    Args:
        data_dir (str): Path to the data directory containing 'images' and 'masks' subdirectories
        
    Returns:
        tuple: Lists of image and mask paths
        
    Raises:
        ValueError: If directory structure is incorrect or number of images doesn't match masks
    """
    data_dir = Path(data_dir)
    images_dir = data_dir / 'images'
    masks_dir = data_dir / 'masks'
    
    if not images_dir.exists() or not masks_dir.exists():
        raise ValueError(f"Data directory structure is incorrect. Expected {data_dir}/images and {data_dir}/masks")
    
    image_paths = sorted(list(images_dir.glob('*')))
    mask_paths = sorted(list(masks_dir.glob('*')))
    
    if len(image_paths) != len(mask_paths):
        raise ValueError(f"Number of images ({len(image_paths)}) doesn't match number of masks ({len(mask_paths)})")
    
    return image_paths, mask_paths

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch.
    
    Args:
        model (nn.Module): The model to train
        train_loader (DataLoader): DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (CPU/GPU)
        
    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    
    for images, masks in tqdm(train_loader, desc='Training'):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    """Validate the model.
    
    Args:
        model (nn.Module): The model to validate
        val_loader (DataLoader): DataLoader for validation data
        criterion: Loss function
        device: Device to validate on (CPU/GPU)
        
    Returns:
        float: Average validation loss
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def main():
    """Main training function.
    
    This function:
    1. Parses command line arguments
    2. Sets up data loading and preprocessing
    3. Creates and trains the model
    4. Saves checkpoints
    5. Handles errors and logging
    """
    parser = argparse.ArgumentParser(description='Train UNet model')
    parser.add_argument('--data_dir', default='data/processed/train', help='Directory containing processed images and masks')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--checkpoint_dir', default='checkpoints', help='Directory to save model checkpoints')
    args = parser.parse_args()
    
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load data
        image_paths, mask_paths = load_data(args.data_dir)
        logger.info(f"Loaded {len(image_paths)} image-mask pairs")
        
        # Create dataset and dataloaders
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        dataset = ImageDataset(image_paths, mask_paths, transform)
        
        # Split into train and validation
        val_size = int(len(dataset) * args.val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        
        # Create model
        model = UNet().to(device)
        
        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        # Create checkpoint directory
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Training loop
        best_val_loss = float('inf')
        for epoch in range(args.epochs):
            logger.info(f"Epoch {epoch+1}/{args.epochs}")
            
            # Train
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            
            # Validate
            val_loss = validate(model, val_loader, criterion, device)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save checkpoint if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = checkpoint_dir / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        raise

if __name__ == '__main__':
    main()
