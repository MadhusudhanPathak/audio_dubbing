"""
Configuration module for Offline Audio Dubbing application.

This module contains all configuration settings and constants used throughout the application.
"""
import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AppConfig:
    """Application configuration settings."""
    
    # Model paths
    WHISPER_EXE_PATH: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Whisper.exe")
    WHISPER_DLL_PATH: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Whisper.dll")
    MODELS_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Models")
    INPUTS_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Inputs")
    OUTPUTS_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Outputs")
    
    # Model directories
    WHISPER_MODELS_DIR: str = os.path.join(MODELS_DIR, "whisper")
    NLLB_MODELS_DIR: str = os.path.join(MODELS_DIR, "nllb")
    XTTS_MODELS_DIR: str = os.path.join(MODELS_DIR, "xtts")
    
    # Supported audio formats
    SUPPORTED_AUDIO_FORMATS: List[str] = ("wav", "mp3", "flac", "m4a", "aac", "ogg", "wma")
    
    # Audio validation settings
    MIN_REF_AUDIO_DURATION: float = 6.0  # seconds
    MAX_REF_AUDIO_DURATION: float = 30.0  # seconds
    
    # Processing settings
    TRANSCRIPTION_TIMEOUT: int = 300  # seconds
    DEFAULT_DEVICE: str = "cuda" if os.environ.get("CUDA_AVAILABLE", "false").lower() == "true" else "cpu"
    
    # Logging settings
    LOG_LEVEL: str = "DEBUG"
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'


# Create a global configuration instance
config = AppConfig()


def get_config() -> AppConfig:
    """Get the application configuration instance."""
    return config