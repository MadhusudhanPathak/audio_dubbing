"""
Configuration module for Offline Audio Dubbing application.

This module contains all configuration settings and constants used throughout the application.
"""
import os
from dataclasses import dataclass
from typing import List


@dataclass
class AppConfig:
    """
    Application configuration settings.

    This class holds all configuration values for the Offline Audio Dubbing application,
    including paths, model directories, audio formats, validation settings, and processing options.

    Attributes:
        PROJECT_ROOT: Absolute path to project root directory
        WHISPER_EXE_PATH: Path to Whisper.exe executable
        WHISPER_DLL_PATH: Path to Whisper.dll library
        MODELS_DIR: Path to models directory
        INPUTS_DIR: Path to inputs directory
        OUTPUTS_DIR: Path to outputs directory
        WHISPER_MODELS_DIR: Path to Whisper models
        NLLB_MODELS_DIR: Path to NLLB models
        XTTS_MODELS_DIR: Path to XTTS models
        SUPPORTED_AUDIO_FORMATS: Tuple of supported audio file extensions
        MIN_REF_AUDIO_DURATION: Minimum reference audio duration in seconds
        MAX_REF_AUDIO_DURATION: Maximum reference audio duration in seconds
        TRANSCRIPTION_TIMEOUT: Transcription operation timeout in seconds
        DEFAULT_DEVICE: Default processing device (CPU or CUDA)
        LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        LOG_FORMAT: Logging message format string
    """

    # Model paths - relative to the project root
    PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    WHISPER_EXE_PATH: str = os.path.join(PROJECT_ROOT, "Whisper.exe")
    WHISPER_DLL_PATH: str = os.path.join(PROJECT_ROOT, "Whisper.dll")
    MODELS_DIR: str = os.path.join(PROJECT_ROOT, "Models")
    INPUTS_DIR: str = os.path.join(PROJECT_ROOT, "Inputs")
    OUTPUTS_DIR: str = os.path.join(PROJECT_ROOT, "Outputs")

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
    """
    Get the application configuration instance.

    Returns:
        The global configuration instance
    """
    return config