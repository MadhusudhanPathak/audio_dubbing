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
    audio_file_path: str
    ref_audio_path: Optional[str] = None
    whisper_model_path: Optional[str] = None
    nllb_model_path: Optional[str] = None
    xtts_model_path: Optional[str] = None
    source_language: Optional[str] = None
    target_languages: Optional[List[str]] = None
    processing_mode: ProcessingMode = ProcessingMode.DUBBED_TRANSLATION


@dataclass
class TranscriptionResult:
    """Result of a transcription operation."""
    text: str
    language: str
    confidence: Optional[float] = None


@dataclass
class TranslationResult:
    """Result of a translation operation."""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str


@dataclass
class VoiceSynthesisResult:
    """Result of a voice synthesis operation."""
    input_text: str
    output_file_path: str
    language: str
    reference_audio_path: Optional[str] = None