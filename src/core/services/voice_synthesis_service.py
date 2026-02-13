from TTS.api import TTS
import torch
import torchaudio
import logging
import os
from pathlib import Path
from typing import Union
from src.utils.common.app_config import get_config


class VoiceSynthesisError(Exception):
    """Custom exception for voice synthesis-related errors."""
    pass


class VoiceCloner:
    def __init__(self, model_path: str):
        """
        Initialize the XTTS-v2 voice cloner with the specified model.

        Args:
            model_path (str): Path to the XTTS-v2 model directory
        
        Raises:
            VoiceSynthesisError: If initialization fails
        """
        # Validate inputs
        if not model_path or not isinstance(model_path, str):
            raise VoiceSynthesisError("Model path must be a non-empty string")

        # Validate model path exists
        if not os.path.exists(model_path):
            raise VoiceSynthesisError(f"XTTS model not found at {model_path}")

        # Check if required model files exist
        # For newer XTTS models (v2.0+), the required files are different
        required_files_new = ['config.json', 'model.pth', 'vocab.json']
        required_files_old = ['config.json', 'model.pth', 'vocab.json', 'speakers.pth', 'language_ids.json']

        # Check for newer format first (without speakers.pth and language_ids.json)
        missing_new = []
        for file in required_files_new:
            if not os.path.exists(os.path.join(model_path, file)):
                missing_new.append(file)

        # If newer format is incomplete, check for older format
        if missing_new:
            missing_old = []
            for file in required_files_old:
                if not os.path.exists(os.path.join(model_path, file)):
                    missing_old.append(file)

            # If both formats are missing required files, raise an error
            if missing_old:
                # Report the minimal set of missing files from the newer format
                raise FileNotFoundError(f"XTTS model is missing required files: {missing_old}")
        else:
            # Newer format is complete
            logging.info("Detected newer XTTS model format (v2.0+)")

        try:
            logging.info(f"Loading XTTS model from {model_path}")
            # Import TTS here to handle potential import issues
            from TTS.api import TTS as TTS_API
            
            # Check if model_path is a directory containing model files or a path to a specific model
            if os.path.isfile(model_path):
                # If model_path is a file, use it directly
                self.tts = TTS_API(
                    model_path=model_path,
                    gpu=torch.cuda.is_available()
                )
            else:
                # If model_path is a directory, verify it's a valid XTTS model directory
                config_path = os.path.join(model_path, "config.json")
                model_path_file = os.path.join(model_path, "model.pth")
                
                if not os.path.exists(config_path):
                    raise FileNotFoundError(f"Config file not found at {config_path}")
                
                if not os.path.exists(model_path_file):
                    raise FileNotFoundError(f"Model file not found at {model_path_file}")
                
                # Try to read and validate the config file
                try:
                    import json
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)
                    
                    # Check if it's an XTTS model by looking for XTTS-specific config elements
                    model_type = config_data.get('model', config_data.get('model_type', ''))
                    if 'xtts' not in model_type.lower():
                        logging.warning(f"Model type '{model_type}' might not be XTTS, proceeding anyway...")
                        
                except json.JSONDecodeError:
                    raise ValueError(f"Config file at {config_path} is not a valid JSON file")
                
                # Load the local XTTS model
                # For local models, the TTS library expects a specific format
                # Try different approaches based on TTS library version
                try:
                    # Approach 1: Load with local model path (this should work for most cases)
                    self.tts = TTS_API(
                        model_path=model_path,
                        config_path=config_path,  # Explicitly specify config path
                        gpu=torch.cuda.is_available()
                    )
                except Exception:
                    # Approach 2: If that fails, try with just the model path
                    try:
                        self.tts = TTS_API(
                            model_path=model_path,
                            gpu=torch.cuda.is_available()
                        )
                    except Exception as load_error:
                        # Approach 3: If both fail, there might be a config format issue
                        logging.error(f"TTS loading failed: {load_error}")
                        # The error might be due to version incompatibility or config format
                        raise
                
            logging.info("XTTS model loaded successfully")
        except ImportError:
            logging.error("Coqui TTS library not found. Please install it with: pip install coqui-tts")
            raise ImportError("Coqui TTS library not found. Please install it with: pip install coqui-tts")
        except Exception as e:
            logging.error(f"Failed to load XTTS model: {str(e)}")
            # Provide guidance to user about potential causes
            logging.error("This error may be caused by:")
            logging.error("- Incompatible model version (model trained with different TTS version)")
            logging.error("- Corrupted config.json file")
            logging.error("- Missing or incorrect model files")
            logging.error("- Version mismatch between TTS library and model")
            raise

    def clone_voice(self, text: str, reference_audio: str, output_path: str, language: str):
        """
        Generate audio with cloned voice from the reference audio.

        Args:
            text (str): Text to synthesize
            reference_audio (str): Path to reference audio file for voice cloning
            output_path (str): Path to save the generated audio
            language (str): Language code for synthesis (e.g., 'en', 'es', 'fr')
        
        Raises:
            VoiceSynthesisError: If voice cloning fails
        """
        # Validate inputs
        if not text or not isinstance(text, str):
            raise VoiceSynthesisError("Text must be a non-empty string")

        if not os.path.exists(reference_audio):
            raise VoiceSynthesisError(f"Reference audio not found at {reference_audio}")

        if not language or not isinstance(language, str):
            raise VoiceSynthesisError("Language must be a non-empty string")

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        try:
            logging.info(f"Starting voice cloning for text length: {len(text)} chars")
            logging.info(f"Reference audio: {reference_audio}")
            logging.info(f"Output path: {output_path}")
            logging.info(f"Original language code: {language}")

            # Convert language code to XTTS format
            # XTTS expects 2-letter language codes (e.g., 'it' for Italian, not 'ita' or 'ita_Latn')
            xtts_language = self._convert_to_xtts_language(language)
            logging.info(f"Converted language code for XTTS: {xtts_language}")

            # Validate that the reference audio is not too long (XTTS performs poorly with very long reference audios)
            duration = self._get_audio_duration(reference_audio)
            if duration > 30:  # More than 30 seconds is typically too long
                logging.warning(f"Reference audio is quite long ({duration}s), this may affect voice cloning quality")
            
            # Synthesize speech with voice cloning
            # Using the correct API for voice cloning with reference audio
            self.tts.tts_to_file(
                text=text,
                speaker_wav=reference_audio,
                file_path=output_path,
                language=xtts_language
            )

            logging.info(f"Voice cloning completed successfully. Output saved to {output_path}")

        except torch.cuda.OutOfMemoryError:
            logging.error("CUDA out of memory during voice cloning. Try using a shorter reference audio or a smaller input text.")
            raise VoiceSynthesisError("CUDA out of memory during voice cloning. Try using a shorter reference audio.")
        except Exception as e:
            logging.error(f"Error during voice cloning: {str(e)}")
            raise VoiceSynthesisError(f"Voice cloning failed: {str(e)}")

    def _convert_to_xtts_language(self, language_code):
        """
        Convert language code to XTTS format (2-letter ISO codes).

        Args:
            language_code (str): Input language code (e.g., 'ita_Latn', 'ita', 'it')

        Returns:
            str: XTTS-compatible 2-letter language code
        """
        # If it's already a 2-letter code that XTTS supports, return as is
        xtts_supported = ['en', 'es', 'fr', 'de', 'it', 'hi']  # Only include our selected languages

        # If it's in the NLLB format like 'eng_Latn', extract the base language code
        if '_' in language_code:
            base_code = language_code.split('_')[0].lower()

            # Special handling for our selected NLLB codes
            nllb_to_xtts = {
                'spa': 'es',      # Spanish
                'fra': 'fr',      # French
                'deu': 'de',      # German
                'ita': 'it',      # Italian
                'hin': 'hi',      # Hindi
            }

            if base_code in nllb_to_xtts:
                xtts_code = nllb_to_xtts[base_code]
                if xtts_code in xtts_supported:
                    return xtts_code
        
        # Extract the base language code (first 2-3 letters)
        if len(language_code) >= 2:
            base_code = language_code[:3].lower() if len(language_code) >= 3 else language_code[:2].lower()

            # Handle special cases for our selected languages
            if language_code.lower().startswith('hi'):
                base_code = 'hi'     # Hindi

            # Check if the base code is supported
            if base_code in xtts_supported:
                return base_code

        # If the 2-letter code isn't supported, try to map our selected languages
        language_mapping = {
            'ita': 'it',      # Italian
            'deu': 'de',      # German
            'fra': 'fr',      # French
            'spa': 'es',      # Spanish
            'hin': 'hi',      # Hindi
        }

        # Check if the full code is in our mapping
        if language_code.lower() in language_mapping:
            mapped_code = language_mapping[language_code.lower()]
            if mapped_code in xtts_supported:
                return mapped_code

        # If it's in the format like 'ita_Latn', extract the first part
        if '_' in language_code:
            base_part = language_code.split('_')[0].lower()
            if base_part in language_mapping:
                mapped_code = language_mapping[base_part]
                if mapped_code in xtts_supported:
                    return mapped_code
            elif base_part in xtts_supported:
                return base_part

        # If we still can't determine, default to English
        logging.warning(f"Language code '{language_code}' not supported by XTTS, defaulting to 'en'")
        return 'en'

    def _get_audio_duration(self, audio_path):
        """
        Get the duration of an audio file in seconds.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            float: Duration in seconds
        """
        try:
            # Try with torchaudio first
            waveform, sample_rate = torchaudio.load(audio_path)
            duration = waveform.shape[1] / sample_rate
            return duration
        except Exception:
            # Fallback to pydub
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_file(audio_path)
                duration = len(audio) / 1000.0  # pydub returns duration in milliseconds
                return duration
            except Exception:
                # If all methods fail, return 0
                logging.warning(f"Could not determine duration of audio file: {audio_path}")
                return 0.0

    def __del__(self):
        """
        Cleanup method to free resources when object is destroyed.
        """
        try:
            if hasattr(self, 'tts'):
                # Attempt to clean up TTS resources if possible
                pass
        except:
            pass  # Ignore errors during cleanup


# Example usage:
if __name__ == "__main__":
    # Example of how to use the VoiceCloner class
    # cloner = VoiceCloner("path/to/xtts/model/")
    # cloner.clone_voice("Hello world", "reference.wav", "output.wav", "en")
    pass