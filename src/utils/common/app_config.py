"""
Configuration module for Offline Audio Dubbing application.

This module contains all configuration settings and constants used throughout the application.
"""
import os
from dataclasses import dataclass
from typing import Tuple


@dataclass
class AppConfig:
    """
    Application configuration settings.

    This class holds all configuration values for the Offline Audio Dubbing application,
    including paths, model directories, audio formats, validation settings, and processing options.

    Attributes:
        PROJECT_ROOT: Absolute path to project root directory
        MODELS_DIR: Path to models directory
        INPUTS_DIR: Path to inputs directory
        OUTPUTS_DIR: Path to outputs directory
        NLLB_MODELS_DIR: Path to NLLB models
        MOSS_AUDIO_MODEL_DIR: Path to MOSS-Audio (STT) models
        MOSS_TTS_MODEL_DIR: Path to MOSS-TTS (voice synthesis) models
        SUPPORTED_AUDIO_FORMATS: List of supported audio file extensions (with leading dot)
        MIN_REF_AUDIO_DURATION: Minimum reference audio duration in seconds
        MAX_REF_AUDIO_DURATION: Maximum reference audio duration in seconds
        TRANSCRIPTION_TIMEOUT: Transcription operation timeout in seconds
        LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        LOG_FORMAT: Logging message format string
    """

    # Model paths - relative to the project root
    PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    MODELS_DIR: str = os.path.join(PROJECT_ROOT, "Models")
    INPUTS_DIR: str = os.path.join(PROJECT_ROOT, "Inputs")
    OUTPUTS_DIR: str = os.path.join(PROJECT_ROOT, "Outputs")

    # Model directories
    NLLB_MODELS_DIR: str = os.path.join(MODELS_DIR, "nllb")
    MOSS_AUDIO_MODEL_DIR: str = os.path.join(MODELS_DIR, "moss-audio")
    MOSS_TTS_MODEL_DIR: str = os.path.join(MODELS_DIR, "moss-tts")

    # Supported audio formats
    SUPPORTED_AUDIO_FORMATS: Tuple[str, ...] = (".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wma")

    # Audio validation settings
    MIN_REF_AUDIO_DURATION: float = 6.0  # seconds
    MAX_REF_AUDIO_DURATION: float = 30.0  # seconds

    # Processing settings
    TRANSCRIPTION_TIMEOUT: int = 300  # seconds

    # Logging settings
    LOG_LEVEL: str = "DEBUG"
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'


config = AppConfig()


def get_config() -> AppConfig:
    """Get the application configuration instance."""
    return config
