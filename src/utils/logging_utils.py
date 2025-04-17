import logging
import os
from datetime import datetime

logger = logging.getLogger("nrse")

def get_log_level(level_str: str) -> int:
    """Convert a logging level name to its corresponding logging constant."""
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    return level_map.get(level_str.upper(), logging.INFO)

def setup_logger(config, log_dir = None):
    """
    Set up a logger with the given configuration.
    
    Args:
        config: Configuration dictionary with logging settings
        
    Returns:
        logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    if not log_dir:
        log_dir = config['training']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"nrse_{timestamp}.log")
    
    # Configure logger
    logger.setLevel(get_log_level(config['logging']['level']))
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create file handler for logging to file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(get_log_level(config['logging']['level']))
    
    # Create console handler for logging to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(get_log_level(config['logging']['console_level']))
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Log initial info
    logger.info(f"Logging level: {config['logging']['level']}")
    logger.info(f"Console logging level: {config['logging']['console_level']}")
    logger.info(f"Log file: {log_file}")
    
    return logger