import os
import argparse
from config.config_utils import get_config
from src.utils.logging_utils import setup_logger, logger

def main():
    parser = argparse.ArgumentParser(description='Train emotion recognition models')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--task', type=str, default='both', choices=['categorical', 'dimensional', 'both'], 
                       help='Which emotion recognition task to train')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    print(args)
    
    # Load config
    config = get_config()
    
    # Set up logging
    setup_logger(config)
    
    if args.task in ['categorical', 'both']:
        logger.info("Starting categorical emotion recognition training")
        from src.train.categorical_emotions import train_categorical_emotions
        categorical_f1 = train_categorical_emotions(config, device=args.device)
        logger.info(f"Categorical training completed with best F1: {categorical_f1:.4f}")
    
    if args.task in ['dimensional', 'both']:
        logger.info("Starting dimensional emotion recognition training")
        from src.train.dimentional_emotions import train_dimensional_emotions
        dimensional_ccc = train_dimensional_emotions(config, device=args.device)
        logger.info(f"Dimensional training completed with best CCC: {dimensional_ccc:.4f}")
    
    logger.info("All training completed!")

if __name__ == "__main__":
    main()