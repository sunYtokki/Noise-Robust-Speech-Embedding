import yaml
from typing import Dict, Any
import argparse
import torch

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        config: Dictionary containing the configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='NRSE: Noise Robust Speech Embeddings')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--task', type=str, default='both', choices=['categorical', 'dimensional', 'both'], 
                       help='Which emotion recognition task to train')
    
    return parser.parse_args()

def get_config():
    """
    Get configuration from YAML file and command line arguments.
    Command line arguments override YAML configuration.
    
    Returns:
        config: Dictionary containing the configuration
    """
    args = parse_args()
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.device is not None:
        config['device'] = args.device
    else:
        # Set default device if not specified
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
        
    if args.epochs is not None:
        config['training']['num_epochs'] = args.epochs
        
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    
    return config