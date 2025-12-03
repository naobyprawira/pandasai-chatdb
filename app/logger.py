import logging
import sys
from logging.handlers import RotatingFileHandler
import os

def _configure_logger(name: str, log_file: str) -> logging.Logger:
    """Internal helper to configure a logger with file and console handlers."""
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler
    try:
        # Use absolute path in /app/logs to ensure it's in the mounted volume
        log_dir = "/app/logs"
        os.makedirs(log_dir, exist_ok=True)
        
        file_path = f"{log_dir}/{log_file}"
        file_handler = RotatingFileHandler(
            file_path, 
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging configured for {name}. Writing to {file_path}")
    except Exception as e:
        sys.stderr.write(f"Failed to setup file logging for {name}: {e}\n")
        
    return logger

def get_app_logger() -> logging.Logger:
    return _configure_logger("app", "app.log")

def get_transform_logger() -> logging.Logger:
    return _configure_logger("transformation", "transformation.log")

def get_chat_logger() -> logging.Logger:
    return _configure_logger("chat", "chat.log")

# Backward compatibility (optional, but good for safety)
def setup_logger(name: str = "app") -> logging.Logger:
    if name == "qa_engine":
        return get_chat_logger()
    elif name == "data_analyzer":
        return get_transform_logger()
    else:
        return get_app_logger()
