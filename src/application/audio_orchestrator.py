"""
Application orchestrator for the Offline Audio Dubbing application.

This module manages the overall workflow and coordinates between different services.
"""

from typing import List, Optional
import logging
from ..data_models.audio_models import (
    AudioProcessingConfig, TranscriptionResult, 
    TranslationResult, VoiceSynthesisResult, ProcessingMode
)
from ..services.transcription_service import Transcriber
from ..services.translation_service import Translator
from ..services.voice_synthesis_service import VoiceCloner
from ..common.helpers import sanitize_filename
from datetime import datetime
import os


class AudioDubbingOrchestrator:
    """
    Orchestrates the complete audio dubbing workflow.
    
    This class manages the entire process from transcription to translation to voice synthesis.
    """
    
    def __init__(self):
        """Initialize the orchestrator."""
        self.logger = logging.getLogger(__name__)
        
    def process_audio(self, config: AudioProcessingConfig) -> bool:
        """
        Process audio according to the specified configuration.
        
        Args:
            config: Configuration for the audio processing
            
        Returns:
            bool: True if processing completed successfully, False otherwise
        """
        try:
            self.logger.info(f"Starting audio processing with mode: {config.processing_mode.value}")
            
            # Ensure output directory exists
            from ..common.helpers import ensure_directory_exists
            ensure_directory_exists("./Outputs")
            
            # Sanitize the output filename to prevent issues
            sanitized_name = sanitize_filename(os.path.splitext(os.path.basename(config.audio_file_path))[0])
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Step 1: Transcription
            transcription_result = self._perform_transcription(
                config.audio_file_path, 
                config.whisper_model_path, 
                config.source_language
            )
            
            if not transcription_result:
                self.logger.error("Transcription failed")
                return False
                
            # If transcription only mode, save and exit
            if config.processing_mode == ProcessingMode.TRANSCRIPTION_ONLY:
                return self._save_transcription_only(transcription_result, sanitized_name, timestamp)
                
            # Step 2: Translation (if in dubbed translation mode)
            if not config.target_languages:
                self.logger.error("No target languages specified for translation mode")
                return False
                
            # Save the original transcription
            transcription_output_path = f"./Outputs/{sanitized_name}_transcription_{timestamp}.txt"
            with open(transcription_output_path, 'w', encoding='utf-8') as f:
                f.write(transcription_result.text)
            self.logger.info(f"Transcription saved to: {transcription_output_path}")
            
            # Perform translation for each target language
            for target_lang in config.target_languages:
                translation_result = self._perform_translation(
                    transcription_result.text,
                    transcription_result.language if config.source_language == "auto" else config.source_language,
                    target_lang,
                    config.nllb_model_path
                )
                
                if not translation_result:
                    self.logger.error(f"Translation to {target_lang} failed")
                    continue
                    
                # Step 3: Voice synthesis
                synthesis_success = self._perform_voice_synthesis(
                    translation_result.translated_text,
                    config.ref_audio_path,
                    target_lang,
                    config.xtts_model_path,
                    sanitized_name,
                    timestamp
                )
                
                if not synthesis_success:
                    self.logger.error(f"Voice synthesis for {target_lang} failed")
                    continue
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Error during audio processing: {str(e)}", exc_info=True)
            return False
    
    def _perform_transcription(self, audio_file_path: str, model_path: str, source_language: str) -> Optional[TranscriptionResult]:
        """
        Perform audio transcription.
        
        Args:
            audio_file_path: Path to the audio file to transcribe
            model_path: Path to the Whisper model
            source_language: Source language for transcription
            
        Returns:
            TranscriptionResult if successful, None otherwise
        """
        try:
            self.logger.info(f"Starting transcription for: {audio_file_path}")
            transcriber = Transcriber(model_path)
            result_dict = transcriber.transcribe(audio_file_path, source_language if source_language != "auto" else None)
            
            result = TranscriptionResult(
                text=result_dict["text"],
                language=result_dict["language"]
            )
            
            self.logger.info(f"Transcription completed. Detected language: {result.language}")
            return result
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {str(e)}", exc_info=True)
            return None
    
    def _perform_translation(self, text: str, source_language: str, target_language: str, model_path: str) -> Optional[TranslationResult]:
        """
        Perform text translation.
        
        Args:
            text: Text to translate
            source_language: Source language code
            target_language: Target language code
            model_path: Path to the NLLB model
            
        Returns:
            TranslationResult if successful, None otherwise
        """
        try:
            self.logger.info(f"Starting translation from {source_language} to {target_language}")
            
            # Map language codes appropriately
            from ..common.helpers import map_language_code
            src_lang_for_nllb = map_language_code(source_language, to_nllb_format=True)
            tgt_lang_for_nllb = map_language_code(target_language, to_nllb_format=True)
            
            translator = Translator(model_path)
            translated_text = translator.translate(text, src_lang_for_nllb, tgt_lang_for_nllb)
            
            result = TranslationResult(
                original_text=text,
                translated_text=translated_text,
                source_language=source_language,
                target_language=target_language
            )
            
            self.logger.info(f"Translation from {source_language} to {target_language} completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Translation failed: {str(e)}", exc_info=True)
            return None
    
    def _perform_voice_synthesis(self, text: str, ref_audio_path: str, language: str, model_path: str, 
                               base_filename: str, timestamp: str) -> bool:
        """
        Perform voice synthesis.
        
        Args:
            text: Text to synthesize
            ref_audio_path: Path to reference audio for voice cloning
            language: Language for synthesis
            model_path: Path to the XTTS model
            base_filename: Base filename for output
            timestamp: Timestamp for output
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.logger.info(f"Starting voice synthesis for language: {language}")
            
            # Map language code for XTTS
            from ..common.helpers import map_language_code
            tgt_lang_for_xtts = map_language_code(language, to_nllb_format=False)
            
            # Create output path
            audio_output_path = f"./Outputs/{base_filename}_dubbed_{language}_{timestamp}.wav"
            
            cloner = VoiceCloner(model_path)
            cloner.clone_voice(text, ref_audio_path, audio_output_path, tgt_lang_for_xtts)
            
            self.logger.info(f"Voice synthesis completed. Output saved to: {audio_output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Voice synthesis failed: {str(e)}", exc_info=True)
            return False
    
    def _save_transcription_only(self, transcription_result: TranscriptionResult, base_filename: str, timestamp: str) -> bool:
        """
        Save transcription result in transcription-only mode.
        
        Args:
            transcription_result: The transcription result to save
            base_filename: Base filename for output
            timestamp: Timestamp for output
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            output_path = f"./Outputs/{base_filename}_transcript_{timestamp}.txt"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(transcription_result.text)
                
            self.logger.info(f"Transcription-only result saved to: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save transcription: {str(e)}", exc_info=True)
            return False