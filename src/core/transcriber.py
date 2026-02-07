import subprocess
import os
import re
import logging
from pathlib import Path
from src.config.app_config import get_config


class Transcriber:
    def __init__(self, model_path):
        """
        Initialize the Whisper transcriber with the specified model.

        Args:
            model_path (str): Path to the Whisper model file (ggml format)
        """
        # Use the configuration for paths
        config = get_config()
        self.whisper_exe = config.WHISPER_EXE_PATH
        self.model_path = model_path

        # Validate model path exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Whisper model not found at {self.model_path}")

        # Validate Whisper executable exists
        if not os.path.exists(self.whisper_exe):
            raise FileNotFoundError(f"Whisper.exe not found at {self.whisper_exe}")
        
        logging.info(f"Initialized Whisper transcriber with model: {self.model_path}")

    def transcribe(self, audio_path, language=None):
        """
        Transcribe audio to text using the Whisper.exe.

        Args:
            audio_path (str): Path to the audio file to transcribe
            language (str, optional): Language code (e.g., 'en', 'es', 'fr')

        Returns:
            dict: Dictionary containing 'text' and 'language' keys
        
        Raises:
            FileNotFoundError: If audio file or model is not found
            ValueError: If audio_path is invalid
            RuntimeError: If transcription fails
        """
        # Validate inputs
        if not audio_path or not isinstance(audio_path, str):
            raise ValueError("Audio path must be a non-empty string")
        
        if language is not None and not isinstance(language, str):
            raise ValueError("Language must be a string or None")
        
        # Validate audio file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found at {audio_path}")

        # Build command arguments for whisper.cpp executable
        cmd = [self.whisper_exe, "-m", self.model_path, "--language"]

        # Add language if specified
        if language and language != "auto":
            # Extract language code from NLLB format if needed (e.g., eng_Latn -> en)
            lang_code = language.split('_')[0][:2] if '_' in language else language[:2]
            cmd.append(lang_code)
        else:
            cmd.append("auto")  # Use auto-detection if no language specified

        # Add output format and suppress verbose output
        cmd.extend(["--output-txt", "--max-len", "1"])

        # Add the audio file as the last argument
        cmd.append(audio_path)
        
        logging.info(f"Starting transcription for audio: {audio_path}, language: {language}")

        try:
            # Run Whisper.exe
            logging.info(f"Starting transcription with command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,  # Don't raise exception on non-zero exit code
                timeout=300  # 5 minute timeout for transcription
            )

            # Check if the command was successful
            if result.returncode != 0:
                logging.error(f"Whisper.exe failed with return code {result.returncode}: {result.stderr}")
                raise RuntimeError(f"Whisper.exe failed with return code {result.returncode}: {result.stderr}")

            # Find the output text file in the same directory as the audio file
            # Whisper creates output files in the same directory as input
            audio_dir = os.path.dirname(audio_path)
            audio_name = os.path.splitext(os.path.basename(audio_path))[0]
            txt_output_path = os.path.join(audio_dir, f"{audio_name}.txt")

            text_result = ""
            if os.path.exists(txt_output_path):
                try:
                    with open(txt_output_path, 'r', encoding='utf-8') as f:
                        text_result = f.read().strip()
                    # Clean up the output file after reading
                    os.remove(txt_output_path)
                    logging.info(f"Successfully read transcription from {txt_output_path}")
                except Exception as e:
                    logging.warning(f"Could not read output file {txt_output_path}: {e}")
                    # If we can't read the file, try to extract from stdout
                    if result.stdout:
                        text_result = self._extract_transcription_from_stdout(result.stdout)

            # If no output file was created, try to extract text from stdout
            if not text_result and result.stdout:
                text_result = self._extract_transcription_from_stdout(result.stdout)

            # If still no text, use stderr as fallback
            if not text_result and result.stderr:
                text_result = result.stderr.strip()

            # Extract language info from output if possible
            detected_language = language if language and language != "auto" else "unknown"

            # If we need to detect language from the output, parse it
            if not language or language == "auto":
                detected_language = self._detect_language_from_output(result.stdout, result.stderr)

            logging.info(f"Transcription completed. Detected language: {detected_language}")
            return {
                "text": text_result,
                "language": detected_language
            }

        except subprocess.TimeoutExpired:
            logging.error("Transcription timed out after 5 minutes")
            raise RuntimeError("Transcription timed out after 5 minutes")
        except Exception as e:
            logging.error(f"Error during transcription: {str(e)}")
            raise

    def _extract_transcription_from_stdout(self, stdout_text):
        """
        Extract clean transcription text from Whisper's stdout output.
        
        Args:
            stdout_text (str): Raw stdout from Whisper.exe
            
        Returns:
            str: Cleaned transcription text
        """
        lines = stdout_text.split('\n')
        filtered_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines and lines with progress indicators
            if line and not line.startswith('[') and ('->' in line or ':' in line):
                # Extract just the text part after the timestamp or identifier
                if '] ' in line:
                    # Format: [timestamp] text
                    text_part = line.split('] ', 1)[1]
                    filtered_lines.append(text_part)
                elif ': ' in line:
                    # Format: identifier: text
                    text_part = line.split(': ', 1)[1]
                    filtered_lines.append(text_part)
                else:
                    # Just add the line if it looks like text
                    if len(line) > 10:  # Likely to be actual text content
                        filtered_lines.append(line)
        
        return ' '.join(filtered_lines).strip()

    def _detect_language_from_output(self, stdout, stderr):
        """
        Detect language from Whisper's output.
        
        Args:
            stdout (str): Standard output from Whisper.exe
            stderr (str): Standard error from Whisper.exe
            
        Returns:
            str: Detected language code
        """
        # Combine outputs to search in both
        combined_output = (stdout + " " + stderr).lower()
        
        # Look for language indicators in the output
        lang_patterns = [
            r"lang[:=]\s*([a-z]{2,3})",
            r"language[:=]\s*([a-z]{2,3})",
            r"detecting language[:=]\s*([a-z]{2,3})",
        ]
        
        for pattern in lang_patterns:
            match = re.search(pattern, combined_output)
            if match:
                return match.group(1)
        
        # Return default if no language detected
        return "unknown"


# Example usage:
if __name__ == "__main__":
    # Example of how to use the Transcriber class
    # transcriber = Transcriber("path/to/whisper/model.ggml")
    # result = transcriber.transcribe("path/to/audio.wav")
    # print(result)
    pass