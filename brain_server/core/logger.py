# Logging configuration for Brain Server
import logging
import os
import sys

def setup_logger(name: str) -> logging.Logger:
    """
    Set up a logger with consistent formatting across the application.
    
    Args:
        name (str): Name of the logger (typically __name__ from calling module)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Get log level from environment (default: INFO)
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    # Avoid duplicate handlers if logger already configured
    if logger.handlers:
        return logger
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, log_level, logging.INFO))
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    return logger
