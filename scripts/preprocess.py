import os
import logging
from PIL import Image
import numpy as np
from pathlib import Path
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_directory(path):
    """Create directory if it doesn't exist.
    
    Args:
        path (str or Path): Path to the directory to create
        
    Returns:
        Path: Path object of the created directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def validate_image(image_path):
    """Validate if the image can be opened and processed.
    
    Args:
        image_path (str or Path): Path to the image file
        
    Returns:
        bool: True if image is valid, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify it's an image
            img = Image.open(image_path)  # Reopen for processing
            return img.mode in ['RGB', 'L', 'RGBA']  # Check if it's a valid color mode
    except Exception as e:
        logger.error(f"Error validating image {image_path}: {str(e)}")
        return False

def create_mask_from_image(image, threshold=128):
    """Create a binary mask from the image using thresholding.
    
    Args:
        image (PIL.Image): Input image
        threshold (int): Threshold value for binary mask (default: 128)
        
    Returns:
        PIL.Image: Binary mask image
    """
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Create binary mask using threshold
    mask = (img_array > threshold).astype(np.uint8) * 255
    
    return Image.fromarray(mask)

def preprocess_images(input_dir, output_dir, target_size=(256, 256)):
    """Process images and create corresponding masks.
    
    This function:
    1. Creates output directories
    2. Finds all image files
    3. Processes each image:
       - Validates the image
       - Resizes to target size
       - Creates a binary mask
       - Saves processed image and mask
    
    Args:
        input_dir (str or Path): Directory containing input images
        output_dir (str or Path): Directory to save processed images and masks
        target_size (tuple): Target size for images (width, height)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    images_dir = create_directory(output_dir / 'images')
    masks_dir = create_directory(output_dir / 'masks')
    
    # Get all image files
    image_files = [f for f in input_dir.glob('*') if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
    
    if not image_files:
        logger.warning(f"No images found in {input_dir}")
        return
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Process each image
    successful = 0
    for img_path in image_files:
        try:
            # Validate image
            if not validate_image(img_path):
                continue
            
            # Open and process image
            with Image.open(img_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize image
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                
                # Save processed image
                output_img_path = images_dir / img_path.name
                img.save(output_img_path, quality=95)
                
                # Create and save mask
                mask = create_mask_from_image(img)
                output_mask_path = masks_dir / img_path.name
                mask.save(output_mask_path)
                
                successful += 1
                
        except Exception as e:
            logger.error(f"Error processing {img_path}: {str(e)}")
            continue
    
    logger.info(f"Successfully processed {successful} out of {len(image_files)} images")

def main():
    """Main preprocessing function.
    
    This function:
    1. Parses command line arguments
    2. Creates necessary directories
    3. Processes images and creates masks
    4. Handles errors and logging
    """
    parser = argparse.ArgumentParser(description='Preprocess images for training')
    parser.add_argument('--input_dir', default='data/raw', help='Input directory containing raw images')
    parser.add_argument('--output_dir', default='data/processed/train', help='Output directory for processed images')
    parser.add_argument('--size', type=int, default=256, help='Target size for images (will be square)')
    args = parser.parse_args()
    
    try:
        # Create directories
        create_directory(args.input_dir)
        create_directory(args.output_dir)
        
        # Process images
        preprocess_images(
            args.input_dir,
            args.output_dir,
            target_size=(args.size, args.size)
        )
        
        logger.info("Preprocessing completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred during preprocessing: {str(e)}")
        raise

if __name__ == '__main__':
    main() 