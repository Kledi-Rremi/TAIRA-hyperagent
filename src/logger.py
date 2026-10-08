"""logger.py — Centralized Logging Configuration

TAIRA (Traffic Accident Intelligent Risk Advisor)
CEN 352 Term Project

Provides structured logging for:
- Error tracking and debugging
- Prediction monitoring
- Critical risk assessment logging
- Performance metrics
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from config import LOG_PREDICTIONS, LOG_CRITICAL_ASSESSMENTS


# Create logs directory if it doesn't exist
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Configure logging format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a configured logger instance.
    
    Args:
        name: Logger name (typically module name)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler for real-time output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler for persistent logs
    log_file = LOG_DIR / f"taira_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


def log_prediction(
    logger: logging.Logger,
    features: Dict[str, Any],
    prediction: str,
    risk_level: str,
    confidence: Optional[float] = None
) -> None:
    """Log a prediction for monitoring and analysis.
    
    Args:
        logger: Logger instance
        features: Input features used for prediction
        prediction: Model prediction result
        risk_level: Risk assessment level
        confidence: Optional confidence score
    """
    if not LOG_PREDICTIONS:
        return
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "prediction": prediction,
        "risk_level": risk_level,
        "features": features,
    }
    
    if confidence is not None:
        log_entry["confidence"] = confidence
    
    logger.info(f"Prediction: {log_entry}")


def log_critical_assessment(
    logger: logging.Logger,
    features: Dict[str, Any],
    prediction: str,
    risk_level: str,
    explanation: str
) -> None:
    """Log CRITICAL risk assessments for quality assurance.
    
    Args:
        logger: Logger instance
        features: Input features
        prediction: Injury severity prediction
        risk_level: Risk level (should be CRITICAL)
        explanation: Explanation text
    """
    if not LOG_CRITICAL_ASSESSMENTS or risk_level != "CRITICAL":
        return
    
    logger.warning(
        f"CRITICAL RISK ASSESSMENT | "
        f"Prediction: {prediction} | "
        f"Weather: {features.get('weather_condition')} | "
        f"Lighting: {features.get('lighting_condition')} | "
        f"Surface: {features.get('roadway_surface_cond')} | "
        f"Explanation: {explanation}"
    )


def log_error(
    logger: logging.Logger,
    error: Exception,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """Log an error with context for debugging.
    
    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Additional context information
    """
    error_msg = f"Error: {type(error).__name__}: {str(error)}"
    if context:
        error_msg += f" | Context: {context}"
    
    logger.error(error_msg, exc_info=True)


# Create module loggers
agent_logger = setup_logger("taira.agent")
model_logger = setup_logger("taira.model")
logic_logger = setup_logger("taira.logic")
data_logger = setup_logger("taira.data")
app_logger = setup_logger("taira.app")
