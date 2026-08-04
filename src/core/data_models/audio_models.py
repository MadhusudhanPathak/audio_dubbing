"""
Data models for the Offline Audio Dubbing application.

This module defines the core data structures used throughout the application.
"""

from dataclasses import dataclass
from typing import Optional, List
from enum import Enum


class ProcessingMode(Enum):
    """Enumeration for different processing modes."""
    TRANSCRIPTION_ONLY = "transcription_only"
    DUBBED_TRANSLATION = "dubbed_translation"


@dataclass
class AudioProcessingConfig:
    """Configuration for audio processing operations."""
    audio_file_path: Optional[str] = None
    text_file_path: Optional[str] = None
    ref_audio_path: Optional[str] = None
    stt_model_path: Optional[str] = None
    nllb_model_path: Optional[str] = None
    tts_model_path: Optional[str] = None
    source_language: Optional[str] = None
    target_languages: Optional[List[str]] = None
    processing_mode: ProcessingMode = ProcessingMode.DUBBED_TRANSLATION