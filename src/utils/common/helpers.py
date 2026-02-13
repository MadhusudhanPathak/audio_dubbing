import os
import glob
from pathlib import Path
from typing import Union, List
import soundfile as sf
import pydub
from pydub import AudioSegment
import logging
from src.utils.app_config import get_config


def scan_model_files(directory: str, extension: Union[str, List[str]] = ".pt") -> List[str]:
    """
    Scan a directory for model files with a specific extension.

    Args:
        directory: Directory path to scan
        extension: File extension(s) to look for (default: ".pt")

    Returns:
        List of model file paths

    Raises:
        ValueError: If directory is not a string or extension is invalid
    """
    if not directory or not isinstance(directory, str):
        raise ValueError("Directory must be a non-empty string")

    if not extension or (not isinstance(extension, str) and not isinstance(extension, list)):
        raise ValueError("Extension must be a string or list of strings")

    if not os.path.exists(directory):
        logging.warning(f"Directory does not exist: {directory}")
        return []

    try:
        files = []
        # Handle multiple extensions if extension is a list
        if isinstance(extension, list):
            for ext in extension:
                if not isinstance(ext, str):
                    logging.warning(f"Skipping invalid extension: {ext}")
                    continue
                pattern = os.path.join(directory, f"*{ext}")
                files.extend(glob.glob(pattern))
        else:
            pattern = os.path.join(directory, f"*{extension}")
            files = glob.glob(pattern)

        logging.info(f"Found {len(files)} model files in {directory}")
        return files
    except Exception as e:
        logging.error(f"Error scanning directory {directory}: {str(e)}")
        return []


def validate_audio_file(file_path: str) -> bool:
    """
    Validate if the given file is a valid audio file.

    Args:
        file_path: Path to the audio file

    Returns:
        True if valid audio file, False otherwise
    """
    if not file_path or not isinstance(file_path, str):
        logging.warning("Invalid file path provided")
        return False

    if not os.path.exists(file_path):
        logging.warning(f"File does not exist: {file_path}")
        return False

    try:
        # Try to load the audio file with soundfile
        data, sr = sf.read(file_path)
        logging.info(f"Validated audio file with soundfile: {file_path}")
        return True
    except Exception as e:
        logging.debug(f"Soundfile validation failed for {file_path}: {str(e)}")
        try:
            # Try with pydub as fallback
            audio = AudioSegment.from_file(file_path)
            logging.info(f"Validated audio file with pydub: {file_path}")
            return True
        except Exception as e2:
            logging.warning(f"Audio file validation failed for {file_path}: {str(e2)}")
            return False


def get_audio_duration(file_path: str) -> float:
    """
    Get the duration of an audio file in seconds.

    Args:
        file_path: Path to the audio file

    Returns:
        Duration in seconds
    """
    if not file_path or not isinstance(file_path, str):
        logging.warning("Invalid file path provided")
        return 0.0

    if not os.path.exists(file_path):
        logging.warning(f"File does not exist: {file_path}")
        return 0.0

    try:
        # Try with soundfile first
        data, sr = sf.read(file_path)
        duration = len(data) / sr
        logging.info(f"Got duration {duration:.2f}s for {file_path} using soundfile")
        return duration
    except Exception as e:
        logging.debug(f"Soundfile duration check failed for {file_path}: {str(e)}")
        try:
            # Fallback to pydub
            audio = AudioSegment.from_file(file_path)
            duration = len(audio) / 1000.0  # pydub returns duration in milliseconds
            logging.info(f"Got duration {duration:.2f}s for {file_path} using pydub")
            return duration
        except Exception as e2:
            logging.error(f"Could not get duration for {file_path}: {str(e2)}")
            return 0.0


def validate_reference_audio_duration(file_path: str, min_duration: int = 6, max_duration: int = 10) -> Tuple[bool, float, str]:
    """
    Validate if the reference audio duration is within the recommended range.

    Args:
        file_path: Path to the reference audio file
        min_duration: Minimum duration in seconds (default: 6)
        max_duration: Maximum duration in seconds (default: 10)

    Returns:
        Tuple of (is_valid, actual_duration, message)
    """
    if not file_path or not isinstance(file_path, str):
        return False, 0.0, "Invalid file path provided"

    if not os.path.exists(file_path):
        return False, 0.0, f"File does not exist: {file_path}"

    duration = get_audio_duration(file_path)

    if duration < min_duration:
        message = f"Reference audio is too short ({duration:.2f}s). Minimum recommended: {min_duration}s."
        logging.warning(message)
        return False, duration, message
    elif duration > max_duration:
        message = f"Reference audio is too long ({duration:.2f}s). Maximum recommended: {max_duration}s."
        logging.warning(message)
        return False, duration, message
    else:
        message = f"Reference audio duration is appropriate ({duration:.2f}s)."
        logging.info(message)
        return True, duration, message


def map_language_code(lang_name: str, to_nllb_format: bool = True) -> str:
    """
    Map between common language names and NLLB language codes.

    Args:
        lang_name: Language name or code
        to_nllb_format: If True, convert to NLLB format; if False, convert from NLLB format

    Returns:
        Mapped language code or name
    """
    # Mapping between common language names and NLLB codes for selected languages only
    # Keys are 2-letter codes, values are NLLB codes
    two_to_nllb = {
        'en': 'eng_Latn',
        'hi': 'hin_Deva',
        'it': 'ita_Latn',
        'de': 'deu_Latn',
        'fr': 'fra_Latn',
        'es': 'spa_Latn',
    }
    
    # Reverse mapping: keys are NLLB codes, values are 2-letter codes
    nllb_to_two = {v: k for k, v in two_to_nllb.items()}

    if not lang_name or not isinstance(lang_name, str):
        logging.warning("Invalid language name provided")
        return lang_name

    if to_nllb_format:
        # Convert to NLLB format
        # If already in NLLB format, return as-is
        if lang_name in nllb_to_two:  # Already in NLLB format
            logging.info(f"Language '{lang_name}' is already in NLLB format, returning as-is")
            return lang_name
        # Otherwise, try to convert from 2-letter format
        result = two_to_nllb.get(lang_name.lower(), lang_name)
        if result == lang_name:
            logging.info(f"No mapping found for language '{lang_name}', returning as-is")
        else:
            logging.info(f"Mapped '{lang_name}' to NLLB format '{result}'")
        return result
    else:
        # Convert to 2-letter format
        # If already in 2-letter format, return as-is
        if lang_name in two_to_nllb:  # Already in 2-letter format
            logging.info(f"Language '{lang_name}' is already in 2-letter format, returning as-is")
            return lang_name
        # Otherwise, try to convert from NLLB format
        result = nllb_to_two.get(lang_name, lang_name)
        if result == lang_name:
            logging.info(f"No reverse mapping found for language '{lang_name}', returning as-is")
        else:
            logging.info(f"Reverse mapped '{lang_name}' to 2-letter format '{result}'")
        return result


def language_code_to_number(lang_code: str) -> int:
    """
    Convert language code to a numeric identifier.

    Args:
        lang_code: Language code (e.g., 'eng_Latn', 'hin_Deva')

    Returns:
        Numeric identifier for the language
    """
    language_numbers = {
        'eng_Latn': 1,
        'hin_Deva': 2,
        'ita_Latn': 3,
        'deu_Latn': 4,
        'fra_Latn': 5,
        'spa_Latn': 6,
    }
    
    return language_numbers.get(lang_code, 0)  # Return 0 for unknown languages


def number_to_language_code(lang_number: int) -> str:
    """
    Convert numeric identifier back to language code.

    Args:
        lang_number: Numeric identifier for the language

    Returns:
        Language code (e.g., 'eng_Latn', 'hin_Deva')
    """
    number_languages = {
        1: 'eng_Latn',
        2: 'hin_Deva',
        3: 'ita_Latn',
        4: 'deu_Latn',
        5: 'fra_Latn',
        6: 'spa_Latn',
    }
    
    return number_languages.get(lang_number, 'eng_Latn')  # Default to English


def ensure_directory_exists(path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure
    """
    if not path or not isinstance(path, str):
        logging.warning("Invalid directory path provided")
        return

    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        logging.info(f"Ensured directory exists: {path}")
    except Exception as e:
        logging.error(f"Failed to create directory {path}: {str(e)}")
        raise


def get_supported_audio_formats() -> List[str]:
    """
    Get a list of supported audio formats.

    Returns:
        List of supported audio file extensions
    """
    formats = ['.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.wma']
    logging.info(f"Supported audio formats: {formats}")
    return formats


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing or replacing invalid characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    if not filename or not isinstance(filename, str):
        return ""

    # Replace invalid characters for file systems
    invalid_chars = '<>:"/\\|?*'
    sanitized = filename
    for char in invalid_chars:
        sanitized = sanitized.replace(char, '_')

    # Remove control characters
    sanitized = ''.join(c for c in sanitized if ord(c) >= 32)

    logging.info(f"Sanitized filename: {filename} -> {sanitized}")
    return sanitized


if __name__ == "__main__":
    # Example usage of utility functions
    # model_files = scan_model_files("../Models/whisper/", ".pt")
    # print(f"Found model files: {model_files}")
    pass