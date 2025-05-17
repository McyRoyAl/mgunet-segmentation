import os
import sys
from argparse import ArgumentParser
import torch

from mgunet.data.dataset import Image2D
from mgunet.mgunet.model import Model
from mgunet.config.default_config import DEVICE_CONFIG

def validate_args(args):
    """Validate command line arguments."""
    if not os.path.exists(args.dataset):
        raise ValueError(f"Dataset path does not exist: {args.dataset}")
    
    if not os.path.exists(args.model_path):
        raise ValueError(f"Model path does not exist: {args.model_path}")
    
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)
    
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str, help='Path to the dataset for prediction')
    parser.add_argument('--results_path', required=True, type=str, help='Path to save prediction results')
    parser.add_argument('--model_path', required=True, type=str, help='Path to the trained model checkpoint')
    parser.add_argument('--device', default=DEVICE_CONFIG['device'], type=str, help='Device to run inference on (cpu/cuda)')
    args = parser.parse_args()

    try:
        validate_args(args)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    try:
        # Load dataset and model
        predict_dataset = Image2D(args.dataset)
        model = torch.load(args.model_path)

        # Initialize model wrapper
        model = Model(model, checkpoint_folder=args.results_path, device=args.device)

        # Run prediction
        model.predict_dataset(predict_dataset, args.results_path)
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()