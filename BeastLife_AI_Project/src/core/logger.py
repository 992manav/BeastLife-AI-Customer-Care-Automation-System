import logging
import logging.handlers
from pathlib import Path
from src.core.config import get_settings

def setup_logger(name: str) -> logging.Logger:
    """Setup logger with file and console handlers."""
    settings = get_settings()
    
    # Create logs directory if not exists
    log_dir = Path(settings.log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(settings.log_level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(settings.log_level)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        settings.log_file,
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(settings.log_level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Create module-level logger
logger = setup_logger(__name__)
