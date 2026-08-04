"""
Application orchestrator for the Offline Audio Dubbing application.

This module manages the overall workflow and coordinates between different services.
"""

import os
import re
import logging
from datetime import datetime
from typing import Optional, Callable

from ..data_models.audio_models import AudioProcessingConfig, ProcessingMode
from ..services.transcription_service import Transcriber
from ..services.translation_service import Translator
from ..services.voice_synthesis_service import VoiceSynthesizer
from ...utils.common.app_config import get_config
from ...utils.model_setup_checker import get_stt_model_path, get_tts_model_path
from ...utils.common.helpers import (
    sanitize_filename, map_language_code, ensure_directory_exists
)


class AudioDubbingOrchestrator:
    """
    Orchestrates the complete audio dubbing workflow.

    This class manages the entire process from transcription to translation to voice synthesis.
    Supports various input types: audio file, transcription text, or translation text.
    """

    def __init__(self, progress_callback: Optional[Callable[[int], None]] = None,
                 status_callback: Optional[Callable[[str], None]] = None,
                 log_callback: Optional[Callable[[str], None]] = None):
        """
        Initialize the orchestrator.

        Args:
            progress_callback: Function to call with progress percentage (0-100)
            status_callback: Function to call with status messages
            log_callback: Function to call with detailed log messages
        """
        self.logger = logging.getLogger(__name__)
        self.progress_callback = progress_callback
        self.status_callback = status_callback
        self.log_callback = log_callback
        self._stop_requested = False

    def _update_progress(self, value: int):
        if self.progress_callback:
            self.progress_callback(value)

    def _update_status(self, message: str):
        if self.status_callback:
            self.status_callback(message)
        self.logger.info(message)

    def _log(self, message: str):
        if self.log_callback:
            self.log_callback(message)
        self.logger.debug(message)

    def stop(self):
        """Request to stop the current processing."""
        self._stop_requested = True

    @staticmethod
    def _output_path(name: str, suffix: str, ts: str, ext: str) -> str:
        """Build an absolute path under the configured Outputs directory."""
        outputs_dir = get_config().OUTPUTS_DIR
        ensure_directory_exists(outputs_dir)
        return os.path.join(outputs_dir, f"{name}_{suffix}_{ts}.{ext}")

    def process(self, config: AudioProcessingConfig, input_type: str = "audio") -> bool:
        """
        Process the request based on input type and configuration.

        Args:
            config: Configuration for processing
            input_type: Type of input ("audio", "transcription", "translation")

        Returns:
            bool: Success status
        """
        try:
            self._stop_requested = False
            ensure_directory_exists(get_config().OUTPUTS_DIR)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_file = config.audio_file_path or config.text_file_path or "dub"
            sanitized_name = sanitize_filename(os.path.splitext(os.path.basename(base_file))[0])

            if input_type == "audio":
                return self._process_audio_input(config, sanitized_name, timestamp)
            elif input_type == "transcription":
                return self._process_transcription_input(config, sanitized_name, timestamp)
            elif input_type == "translation":
                return self._process_translation_input(config, sanitized_name, timestamp)
            else:
                self._update_status(f"Error: Unknown input type {input_type}")
                return False

        except Exception as e:
            self._update_status(f"Error: {str(e)}")
            self.logger.error(f"Processing failed: {str(e)}", exc_info=True)
            return False

    def _process_audio_input(self, config: AudioProcessingConfig, name: str, ts: str) -> bool:
        if not config.audio_file_path or not os.path.exists(config.audio_file_path):
            raise ValueError(f"Audio file not found: {config.audio_file_path}")

        self._update_status("Loading MOSS-Audio model...")
        stt_model_path = get_stt_model_path()
        transcriber = Transcriber(stt_model_path)
        try:
            self._update_progress(10)

            if self._stop_requested: return False

            self._update_status("Transcribing audio...")
            text = transcriber.transcribe(config.audio_file_path, config.source_language)
            self._update_progress(30)

            if config.processing_mode == ProcessingMode.TRANSCRIPTION_ONLY:
                output_path = self._output_path(name, "transcript", ts, "txt")
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                self._update_status(f"Transcription saved to {output_path}")
                self._update_progress(100)
                return True

            cleaned_text = self._clean_transcription(text)
        finally:
            transcriber.unload()

        return self._translate_and_dub(config, cleaned_text, config.source_language, name, ts, start_progress=40)

    def _process_transcription_input(self, config: AudioProcessingConfig, name: str, ts: str) -> bool:
        if not config.text_file_path or not os.path.exists(config.text_file_path):
            raise ValueError(f"Transcription text file not found: {config.text_file_path}")

        self._update_status("Loading transcription text...")
        with open(config.text_file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        self._update_progress(20)

        src_lang = config.source_language if config.source_language != "auto" else "eng_Latn"
        return self._translate_and_dub(config, text, src_lang, name, ts, start_progress=40)

    def _process_translation_input(self, config: AudioProcessingConfig, name: str, ts: str) -> bool:
        if not config.text_file_path or not os.path.exists(config.text_file_path):
            raise ValueError(f"Translation text file not found: {config.text_file_path}")

        self._update_status("Loading translation text...")
        with open(config.text_file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        self._update_progress(20)

        target_lang = config.target_languages[0] if config.target_languages else "eng_Latn"
        return self._dub_only(config, text, target_lang, name, ts)

    def _translate_and_dub(self, config: AudioProcessingConfig, text: str, src_lang: str,
                           name: str, ts: str, start_progress: int) -> bool:
        self._update_status("Loading NLLB translator...")
        translator = Translator(config.nllb_model_path)

        num_langs = len(config.target_languages)
        progress_per_lang = (100 - start_progress) // num_langs if num_langs > 0 else 0

        synthesizer = None
        try:
            for i, tgt_lang in enumerate(config.target_languages):
                if self._stop_requested: return False

                self._update_status(f"Translating to {tgt_lang}...")
                src_nllb = map_language_code(src_lang, to_nllb_format=True)
                tgt_nllb = map_language_code(tgt_lang, to_nllb_format=True)

                translated = translator.translate(text, src_nllb, tgt_nllb)

                txt_out = self._output_path(name, f"translation_{tgt_lang}", ts, "txt")
                with open(txt_out, 'w', encoding='utf-8') as f:
                    f.write(translated)

                if self._stop_requested: return False

                self._update_status(f"Generating audio for {tgt_lang}...")
                if synthesizer is None:
                    tts_model_path = get_tts_model_path()
                    synthesizer = VoiceSynthesizer(tts_model_path)

                audio_out = self._output_path(name, f"dubbed_{tgt_lang}", ts, "wav")
                synthesizer.synthesize(
                    text=translated,
                    output_path=audio_out,
                    reference_audio_path=config.ref_audio_path,
                )

                self._update_progress(start_progress + (i + 1) * progress_per_lang)

            self._update_status("Processing completed successfully!")
            self._update_progress(100)
            return True
        finally:
            translator.unload()
            if synthesizer:
                synthesizer.unload()

    def _dub_only(self, config: AudioProcessingConfig, text: str, tgt_lang: str, name: str, ts: str) -> bool:
        self._update_status("Loading MOSS-TTS model...")
        tts_model_path = get_tts_model_path()
        synthesizer = VoiceSynthesizer(tts_model_path)
        try:
            self._update_progress(50)

            if self._stop_requested: return False

            self._update_status(f"Generating audio for {tgt_lang}...")
            audio_out = self._output_path(name, f"dubbed_{tgt_lang}", ts, "wav")
            synthesizer.synthesize(
                text=text,
                output_path=audio_out,
                reference_audio_path=config.ref_audio_path,
            )

            self._update_status(f"Audio saved to {audio_out}")
            self._update_progress(100)
            return True
        finally:
            synthesizer.unload()

    def _clean_transcription(self, raw_text: str) -> str:
        """Extract plain text from transcription output (removing timestamps if any)."""
        lines = raw_text.split('\n')
        text_parts = []
        seen_words = set()

        for line in lines:
            line = line.strip()
            if '-->' in line and '] ' in line:
                parts = line.split('] ', 1)
                if len(parts) > 1:
                    text_part = parts[1].strip()
                    if text_part and not text_part.startswith('['):
                        words = text_part.split()
                        unique_words = [w for w in words if w.lower() not in seen_words]
                        for w in unique_words: seen_words.add(w.lower())
                        if unique_words:
                            text_parts.append(' '.join(unique_words))

        cleaned = ' '.join(text_parts).strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned if cleaned else raw_text
