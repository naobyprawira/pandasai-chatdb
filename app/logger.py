import logging
import sys
from logging.handlers import RotatingFileHandler
import os

def setup_logger(name: str = "app") -> logging.Logger:
    """
    Setup logger with file and console handlers.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # If logger already has handlers, assume it's already configured
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console Handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler
    try:
        # Use absolute path in /app to ensure it's in the mounted volume
        log_file = "/app/app.log"
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging configured. Writing to {log_file}")
    except Exception as e:
        # Print to stderr to ensure it's seen
        sys.stderr.write(f"Failed to setup file logging: {e}\n")
        
    return logger
