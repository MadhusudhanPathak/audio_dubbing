"""
transcription_service.py
Speech-to-text using OpenMOSS MOSS-Audio (replaces Whisper).
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torchaudio
from transformers import AutoModel, AutoProcessor

from src.core.services.base_model_service import HFModelService

logger = logging.getLogger(__name__)


class Transcriber(HFModelService):
    """
    Drop-in replacement for the old Whisper-based Transcriber.
    Uses OpenMOSS MOSS-Audio-{4B|8B}-Instruct for speech recognition.
    """

    PROMPT_TRANSCRIBE      = "Transcribe the speech in this audio."
    PROMPT_TRANSCRIBE_LANG = "Transcribe the speech in this audio in {language}."
    PROMPT_TIMESTAMP_ASR   = (
        "Transcribe the speech in this audio with word-level timestamps."
    )

    def __init__(self, model_path: str, device: Optional[str] = None) -> None:
        super().__init__(model_path, device)

        logger.info(f"Loading MOSS-Audio from {model_path} on {self.device}")

        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()

        logger.info("MOSS-Audio loaded successfully.")

    def transcribe(
        self,
        audio_path: str,
        language:   Optional[str] = None,
        timestamps: bool = False,
    ) -> str:
        """
        Transcribe speech from an audio file.

        Args:
            audio_path: Path to the input audio file.
            language:   Optional target language name (e.g. "English", "French").
                        If None, MOSS-Audio auto-detects.
            timestamps: If True, requests word-level timestamp output.

        Returns:
            Transcription string.
        """
        waveform, sr = torchaudio.load(str(audio_path))

        if timestamps:
            prompt = self.PROMPT_TIMESTAMP_ASR
        elif language:
            prompt = self.PROMPT_TRANSCRIBE_LANG.format(language=language)
        else:
            prompt = self.PROMPT_TRANSCRIBE

        inputs = self.processor(
            audio=waveform,
            sampling_rate=sr,
            text=prompt,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=2048)

        transcription = self.processor.decode(
            output_ids[0], skip_special_tokens=True
        ).strip()

        logger.info(f"Transcription complete ({len(transcription)} chars)")
        return transcription

    def get_supported_languages(self) -> list[str]:
        """
        MOSS-Audio supports all major world languages via its Qwen3 backbone.
        This list covers the languages tested for ASR quality in the official eval.
        """
        return [
            "Chinese", "English", "French", "German", "Spanish",
            "Japanese", "Korean", "Portuguese", "Arabic", "Russian",
            "Italian", "Dutch", "Polish", "Turkish", "Hindi",
        ]
