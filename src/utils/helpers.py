import os
import glob
from pathlib import Path
import soundfile as sf
import pydub
from pydub import AudioSegment
import logging
from src.config.app_config import get_config


def scan_model_files(directory, extension=".pt"):
    """
    Scan a directory for model files with a specific extension.

    Args:
        directory (str): Directory path to scan
        extension (str or list): File extension(s) to look for (default: ".pt")

    Returns:
        list: List of model file paths
    
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
        # Handle multiple extensions if extension is a list
        if isinstance(extension, list):
            files = []
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


def validate_audio_file(file_path):
    """
    Validate if the given file is a valid audio file.

    Args:
        file_path (str): Path to the audio file

    Returns:
        bool: True if valid audio file, False otherwise
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


def get_audio_duration(file_path):
    """
    Get the duration of an audio file in seconds.

    Args:
        file_path (str): Path to the audio file

    Returns:
        float: Duration in seconds
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


def validate_reference_audio_duration(file_path, min_duration=6, max_duration=10):
    """
    Validate if the reference audio duration is within the recommended range.

    Args:
        file_path (str): Path to the reference audio file
        min_duration (int): Minimum duration in seconds (default: 6)
        max_duration (int): Maximum duration in seconds (default: 10)

    Returns:
        tuple: (bool: is_valid, float: actual_duration, str: message)
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


def map_language_code(lang_name, to_nllb_format=True):
    """
    Map between common language names and NLLB language codes.

    Args:
        lang_name (str): Language name or code
        to_nllb_format (bool): If True, convert to NLLB format; if False, convert from NLLB format

    Returns:
        str: Mapped language code or name
    """
    # Mapping between common language names and NLLB codes
    lang_map = {
        'en': 'eng_Latn',
        'eng_Latn': 'en',
        'es': 'spa_Latn',
        'spa_Latn': 'es',
        'fr': 'fra_Latn',
        'fra_Latn': 'fr',
        'de': 'deu_Latn',
        'deu_Latn': 'de',
        'it': 'ita_Latn',
        'ita_Latn': 'it',
        'pt': 'por_Latn',
        'por_Latn': 'pt',
        'ru': 'rus_Cyrl',
        'rus_Cyrl': 'ru',
        'zh': 'zho_Hans',
        'zho_Hans': 'zh',
        'ja': 'jpn_Jpan',
        'jpn_Jpan': 'ja',
        'ko': 'kor_Hang',
        'kor_Hang': 'ko',
        'ar': 'ara_Arab',
        'ara_Arab': 'ar',
        'hi': 'hin_Deva',
        'hin_Deva': 'hi',
    }

    if not lang_name or not isinstance(lang_name, str):
        logging.warning("Invalid language name provided")
        return lang_name

    if to_nllb_format:
        result = lang_map.get(lang_name.lower(), lang_name)
        if result == lang_name:
            logging.info(f"No mapping found for language '{lang_name}', returning as-is")
        else:
            logging.info(f"Mapped '{lang_name}' to '{result}'")
        return result
    else:
        result = lang_map.get(lang_name, lang_name)
        if result == lang_name:
            logging.info(f"No reverse mapping found for language '{lang_name}', returning as-is")
        else:
            logging.info(f"Reverse mapped '{lang_name}' to '{result}'")
        return result


def ensure_directory_exists(path):
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        path (str): Directory path to ensure
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


def get_supported_audio_formats():
    """
    Get a list of supported audio formats.

    Returns:
        list: List of supported audio file extensions
    """
    formats = ['.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.wma']
    logging.info(f"Supported audio formats: {formats}")
    return formats


def sanitize_filename(filename):
    """
    Sanitize filename by removing or replacing invalid characters.

    Args:
        filename (str): Original filename

    Returns:
        str: Sanitized filename
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