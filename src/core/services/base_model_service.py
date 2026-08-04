"""
base_model_service.py
Shared device/dtype resolution and unload lifecycle for local HuggingFace-backed
STT/TTS model wrappers (Transcriber, VoiceSynthesizer).
"""
from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class HFModelService:
    """Common lifecycle base for services that wrap a local HuggingFace model + processor."""

    def __init__(self, model_path: str, device: Optional[str] = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model_path = model_path
        self.model = None
        self.processor = None

    def unload(self) -> None:
        """Free GPU memory after use. Safe to call more than once."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info(f"{type(self).__name__} unloaded.")
