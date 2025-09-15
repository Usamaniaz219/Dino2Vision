import logging 
from rich.logging import RichHandler
from rich.console import Console

RICH_AVAILABLE = True


def setup_logging(log_level=logging.INFO, log_file=None):
    """Setup logging configuration"""
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    if RICH_AVAILABLE:
        console_handler = RichHandler(rich_tracebacks=True)
        console_handler.setFormatter(logging.Formatter('%(message)s'))
    else:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
