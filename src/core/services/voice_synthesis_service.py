"""
voice_synthesis_service.py
Voice synthesis using OpenMOSS MOSS-TTS (replaces XTTS-v2).
"""
from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from transformers import AutoModel, AutoProcessor

from src.core.services.base_model_service import HFModelService

logger = logging.getLogger(__name__)

_sdp_backends_patched = False


def _patch_sdp_backends() -> None:
    """Configure CUDA SDP attention backends required by MOSS-TTS. Global torch
    state, so this only needs to run once per process."""
    global _sdp_backends_patched
    if _sdp_backends_patched or not torch.cuda.is_available():
        return
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    _sdp_backends_patched = True


def _resolve_attn(device: str, dtype: torch.dtype) -> str:
    if device == "cuda" and dtype in {torch.float16, torch.bfloat16}:
        if importlib.util.find_spec("flash_attn") is not None:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                return "flash_attention_2"
        return "sdpa"
    return "eager"


class VoiceSynthesizer(HFModelService):
    def __init__(self, model_path: str, device: Optional[str] = None) -> None:
        _patch_sdp_backends()

        super().__init__(model_path, device)

        logger.info(f"Loading MOSS-TTS from {model_path} on {self.device}")

        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )
        self.processor.audio_tokenizer = self.processor.audio_tokenizer.to(self.device)

        attn = _resolve_attn(self.device, self.dtype)
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation=attn,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()
        logger.info("MOSS-TTS loaded successfully.")

    def synthesize(
        self,
        text:                 str,
        output_path:          str,
        reference_audio_path: Optional[str] = None,
    ) -> str:
        if reference_audio_path:
            conversation = [
                self.processor.build_user_message(
                    text=text, reference=[str(reference_audio_path)]
                )
            ]
        else:
            logger.warning("No reference audio — MOSS-TTS will use a random timbre.")
            conversation = [self.processor.build_user_message(text=text)]

        batch = self.processor(conversation, mode="generation")
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids      = batch["input_ids"].to(self.device),
                attention_mask = batch["attention_mask"].to(self.device),
                max_new_tokens = 4096,
            )

        for message in self.processor.decode(outputs):
            audio = message.audio_codes_list[0]
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(
                output_path,
                audio.unsqueeze(0),
                self.processor.model_config.sampling_rate,
            )
            logger.info(f"Saved audio to {output_path}")
            return output_path

        raise RuntimeError("MOSS-TTS decode produced no output.")

    def get_supported_languages(self) -> list[str]:
        return [
            "zh", "en", "de", "es", "fr", "ja", "it", "hu",
            "ko", "ru", "fa", "ar", "pl", "pt", "cs", "da",
            "sv", "el", "tr", "no",
        ]
