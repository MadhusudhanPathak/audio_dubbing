import torch
import os
import logging
from pathlib import Path
from typing import Optional


class VoiceSynthesisError(Exception):
    """Custom exception for voice synthesis-related errors."""
    pass


class VoiceCloner:
    """XTTS-v2 based voice synthesis and cloning service."""

    def __init__(self, model_path: str):
        if not model_path or not os.path.exists(model_path):
            raise VoiceSynthesisError(f"XTTS model not found at {model_path}")

        try:
            from TTS.api import TTS
            self.tts = TTS(model_path=model_path, gpu=torch.cuda.is_available())
        except Exception as e:
            raise VoiceSynthesisError(f"Failed to load XTTS model: {str(e)}")

    def clone_voice(self, text: str, reference_audio: str, output_path: str, language: str):
        """
        Synthesize audio with cloned voice.
        
        Args:
            text: Text to synthesize
            reference_audio: Path to reference audio for cloning
            output_path: Path to save result
            language: Target language (2-letter code)
        """
        if not text.strip(): return
        
        if not os.path.exists(reference_audio):
            raise VoiceSynthesisError(f"Reference audio missing: {reference_audio}")

        try:
            # Map NLLB codes to XTTS codes if needed
            xtts_lang = self._map_lang(language)
            
            self.tts.tts_to_file(
                text=text,
                speaker_wav=reference_audio,
                file_path=output_path,
                language=xtts_lang
            )
        except Exception as e:
            raise VoiceSynthesisError(f"Voice synthesis failed: {str(e)}")

    def _map_lang(self, lang: str) -> str:
        mapping = {'spa': 'es', 'fra': 'fr', 'deu': 'de', 'ita': 'it', 'hin': 'hi'}
        base = lang.split('_')[0].lower() if '_' in lang else lang[:3].lower()
        return mapping.get(base, base[:2])


