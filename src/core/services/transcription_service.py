import subprocess
import os
import re
import logging
from typing import Optional, Dict
from src.utils.common.app_config import get_config


class TranscriptionError(Exception):
    """Custom exception for transcription-related errors."""
    pass


class Transcriber:
    """Whisper-based audio transcription service."""

    def __init__(self, model_path: str):
        config = get_config()
        self.whisper_exe = config.WHISPER_EXE_PATH
        self.model_path = model_path

        if not model_path or not os.path.exists(model_path):
            raise TranscriptionError(f"Whisper model not found at {model_path}")

        if not os.path.exists(self.whisper_exe):
            logging.warning(f"Whisper.exe not found at {self.whisper_exe}")

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> Dict[str, str]:
        """
        Transcribe audio to text.
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'eng_Latn')
            
        Returns:
            Dict with 'text' and 'language'
        """
        if not os.path.exists(audio_path):
            raise TranscriptionError(f"Audio file not found: {audio_path}")

        if not os.path.exists(self.whisper_exe):
            raise TranscriptionError("Whisper.exe missing from project root")

        cmd = [self.whisper_exe, "-m", self.model_path, "--language"]
        
        if language and language != "auto":
            lang_code = language.split('_')[0][:2] if '_' in language else language[:2]
            cmd.append(lang_code)
        else:
            cmd.append("auto")

        cmd.extend(["--output-txt", "--max-len", "1", audio_path])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                raise RuntimeError(f"Whisper.exe failed: {result.stderr}")

            audio_dir = os.path.dirname(audio_path)
            audio_name = os.path.splitext(os.path.basename(audio_path))[0]
            txt_output_path = os.path.join(audio_dir, f"{audio_name}.txt")

            text_result = ""
            if os.path.exists(txt_output_path):
                with open(txt_output_path, 'r', encoding='utf-8') as f:
                    text_result = f.read().strip()
                os.remove(txt_output_path)

            if not text_result and result.stdout:
                text_result = self._extract_text(result.stdout)

            detected_lang = language if language and language != "auto" else self._detect_lang(result.stdout, result.stderr)

            return {"text": text_result or result.stderr.strip(), "language": detected_lang}

        except subprocess.TimeoutExpired:
            raise TranscriptionError("Transcription timed out")
        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {str(e)}")

    def _extract_text(self, stdout: str) -> str:
        lines = []
        for line in stdout.split('\n'):
            line = line.strip()
            if line and not line.startswith('['):
                if '] ' in line: lines.append(line.split('] ', 1)[1])
                elif ': ' in line: lines.append(line.split(': ', 1)[1])
                elif len(line) > 10: lines.append(line)
        return ' '.join(lines).strip()

    def _detect_lang(self, stdout: str, stderr: str) -> str:
        combined = (stdout + " " + stderr).lower()
        match = re.search(r"(?:lang|language)[:=]\s*([a-z]{2,3})", combined)
        return match.group(1) if match else "unknown"



