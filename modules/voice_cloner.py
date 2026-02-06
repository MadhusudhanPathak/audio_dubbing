from TTS.api import TTS
import torch
import torchaudio
import logging
import os
from pathlib import Path


class VoiceCloner:
    def __init__(self, model_path):
        """
        Initialize the XTTS-v2 voice cloner with the specified model.

        Args:
            model_path (str): Path to the XTTS-v2 model directory
        """
        # Validate model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"XTTS model not found at {model_path}")
        
        try:
            logging.info(f"Loading XTTS model from {model_path}")
            # Initialize the TTS model with proper device detection
            self.tts = TTS(model_path, gpu=torch.cuda.is_available())
            logging.info("XTTS model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load XTTS model: {str(e)}")
            raise

    def clone_voice(self, text, reference_audio, output_path, language):
        """
        Generate audio with cloned voice from the reference audio.

        Args:
            text (str): Text to synthesize
            reference_audio (str): Path to reference audio file for voice cloning
            output_path (str): Path to save the generated audio
            language (str): Language code for synthesis (e.g., 'en', 'es', 'fr')
        """
        # Validate inputs
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        
        if not os.path.exists(reference_audio):
            raise FileNotFoundError(f"Reference audio not found at {reference_audio}")
        
        if not language or not isinstance(language, str):
            raise ValueError("Language must be a non-empty string")
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            logging.info(f"Starting voice cloning for text length: {len(text)} chars")
            logging.info(f"Reference audio: {reference_audio}")
            logging.info(f"Output path: {output_path}")
            logging.info(f"Language: {language}")
            
            # Synthesize speech with voice cloning
            self.tts.tts_to_file(
                text=text,
                speaker_wav=reference_audio,
                file_path=output_path,
                language=language
            )
            
            logging.info(f"Voice cloning completed successfully. Output saved to {output_path}")
            
        except Exception as e:
            logging.error(f"Error during voice cloning: {str(e)}")
            raise

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