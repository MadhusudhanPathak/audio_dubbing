import os
import glob
from pathlib import Path
from typing import Union, List, Tuple
import soundfile as sf
import pydub
from pydub import AudioSegment
import logging
from src.utils.common.app_config import get_config


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


def get_nllb_languages() -> dict:
    """
    Get all supported NLLB-200 languages.

    Returns:
        Dictionary mapping language names to NLLB codes
    """
    return {
        "Acehnese (Arabic script)": "ace_Arab",
        "Acehnese (Latin script)": "ace_Latn",
        "Mesopotamian Arabic": "acm_Arab",
        "Ta'izzi-Adeni Arabic": "acq_Arab",
        "Tunisian Arabic": "aeb_Arab",
        "Afrikaans": "afr_Latn",
        "Levantine Arabic": "apc_Arab",
        "Standard Arabic": "arb_Arab",
        "Arabic": "arb_Arab",
        "Najdi Arabic": "ars_Arab",
        "Moroccan Arabic": "ary_Arab",
        "Egyptian Arabic": "arz_Arab",
        "Assamese": "asm_Beng",
        "Asturian": "ast_Latn",
        "Awadhi": "awa_Deva",
        "Central Aymara": "ayr_Latn",
        "South Aymara": "ayr_Latn",
        "Azerbaijani": "azj_Latn",
        "Bashkir": "bak_Cyrl",
        "Bambara": "bam_Latn",
        "Balinese": "ban_Latn",
        "Belarusian": "bel_Cyrl",
        "Bengali": "ben_Beng",
        "Bhojpuri": "bho_Deva",
        "Banjar (Arabic script)": "bjn_Arab",
        "Banjar (Latin script)": "bjn_Latn",
        "Standard Tibetan": "bod_Tibt",
        "Tibetan": "bod_Tibt",
        "Bosnian": "bos_Latn",
        "Buginese": "bug_Latn",
        "Bulgarian": "bul_Cyrl",
        "Catalan": "cat_Latn",
        "Cebuano": "ceb_Latn",
        "Czech": "ces_Latn",
        "Chokwe": "cjk_Latn",
        "Central Kurdish": "ckb_Arab",
        "Crimean Tatar": "crh_Latn",
        "Welsh": "cym_Latn",
        "Danish": "dan_Latn",
        "German": "deu_Latn",
        "Southwestern Dinka": "dik_Latn",
        "Dyula": "dyu_Latn",
        "Dzongkha": "dzo_Tibt",
        "Greek": "ell_Grek",
        "English": "eng_Latn",
        "Esperanto": "epo_Latn",
        "Estonian": "est_Latn",
        "Basque": "eus_Latn",
        "Ewe": "ewe_Latn",
        "Faroese": "fao_Latn",
        "Fijian": "fij_Latn",
        "Finnish": "fin_Latn",
        "Fon": "fon_Latn",
        "French": "fra_Latn",
        "Friulian": "fur_Latn",
        "Nigerian Fulfulde": "fuv_Latn",
        "Scottish Gaelic": "gla_Latn",
        "Irish": "gle_Latn",
        "Galician": "glg_Latn",
        "Guarani": "grn_Latn",
        "Gujarati": "guj_Gujr",
        "Haitian Creole": "hat_Latn",
        "Hausa": "hau_Latn",
        "Hebrew": "heb_Hebr",
        "Hindi": "hin_Deva",
        "Chhattisgarhi": "hne_Deva",
        "Croatian": "hrv_Latn",
        "Hungarian": "hun_Latn",
        "Armenian": "hye_Armn",
        "Igbo": "ibo_Latn",
        "Ilocano": "ilo_Latn",
        "Indonesian": "ind_Latn",
        "Icelandic": "isl_Latn",
        "Italian": "ita_Latn",
        "Javanese": "jav_Latn",
        "Japanese": "jpn_Jpan",
        "Kabyle": "kab_Latn",
        "Jingpho": "kac_Latn",
        "Kamba": "kam_Latn",
        "Kannada": "kan_Knda",
        "Kashmiri (Arabic script)": "kas_Arab",
        "Kashmiri (Devanagari script)": "kas_Deva",
        "Georgian": "kat_Geor",
        "Central Kanuri (Arabic script)": "knc_Arab",
        "Central Kanuri (Latin script)": "knc_Latn",
        "Kazakh": "kaz_Cyrl",
        "Kabiyè": "kbp_Latn",
        "Kabuverdianu": "kea_Latn",
        "Khmer": "khm_Khmr",
        "Kikuyu": "kik_Latn",
        "Kinyarwanda": "kin_Latn",
        "Kyrgyz": "kir_Cyrl",
        "Kimbundu": "kmb_Latn",
        "Northern Kurdish": "kmr_Latn",
        "Kikongo": "kon_Latn",
        "Korean": "kor_Hang",
        "Lao": "lao_Laoo",
        "Ligurian": "lij_Latn",
        "Limburgish": "lim_Latn",
        "Lingala": "lin_Latn",
        "Lithuanian": "lit_Latn",
        "Lombard": "lmo_Latn",
        "Latgalian": "ltg_Latn",
        "Luxembourgish": "ltz_Latn",
        "Luba-Kasai": "lua_Latn",
        "Ganda": "lug_Latn",
        "Luo": "luo_Latn",
        "Mizo": "lus_Latn",
        "Standard Latvian": "lvs_Latn",
        "Latvian": "lvs_Latn",
        "Magahi": "mag_Deva",
        "Maithili": "mai_Deva",
        "Malayalam": "mal_Mlym",
        "Marathi": "mar_Deva",
        "Minangkabau (Arabic script)": "min_Arab",
        "Minangkabau (Latin script)": "min_Latn",
        "Macedonian": "mkd_Cyrl",
        "Plateau Mogul": "mks_Latn",
        "Maltese": "mlt_Latn",
        "Meitei (Bengali script)": "mni_Beng",
        "Meitei (Meitei script)": "mni_Mtei",
        "Khmer": "khm_Khmr",
        "Moselle Franconian": "moslfr_Latn",
        "Moroccan Arabic": "mya_Mymr",
        "Burmese": "mya_Mymr",
        "Dutch": "nld_Latn",
        "Norwegian Nynorsk": "nno_Latn",
        "Norwegian Bokmål": "nob_Latn",
        "Nepali": "npi_Deva",
        "Northern Sotho": "nso_Latn",
        "Nuer": "nus_Latn",
        "Nyanja": "nya_Latn",
        "Occitan": "oci_Latn",
        "West Central Oromo": "gaz_Latn",
        "Oromo": "orm_Latn",
        "Odia": "ory_Orya",
        "Pangasinan": "pag_Latn",
        "Eastern Panjabi": "pan_Guru",
        "Papiamento": "pap_Latn",
        "Western Persian": "pes_Arab",
        "Polish": "pol_Latn",
        "Portuguese": "por_Latn",
        "Dari": "prs_Arab",
        "Southern Pashto": "pbt_Arab",
        "Ayacucho Quechua": "quy_Latn",
        "Quechua": "quy_Latn",
        "Romanian": "ron_Latn",
        "Rundi": "run_Latn",
        "Russian": "rus_Cyrl",
        "Sango": "sag_Latn",
        "Sanskrit": "san_Deva",
        "Santali": "sat_Olck",
        "Sicilian": "scn_Latn",
        "Shan": "shn_Mymr",
        "Sinhala": "sin_Sinh",
        "Slovak": "slk_Latn",
        "Slovenian": "slv_Latn",
        "Samoan": "smo_Latn",
        "Shona": "sna_Latn",
        "Sindhi": "snd_Arab",
        "Somali": "som_Latn",
        "Southern Sotho": "sot_Latn",
        "Spanish": "spa_Latn",
        "Tosk Albanian": "als_Latn",
        "Sardinian": "srd_Latn",
        "Serbian": "srp_Cyrl",
        "Swati": "ssw_Latn",
        "Sundanese": "sun_Latn",
        "Swedish": "swe_Latn",
        "Swahili": "swh_Latn",
        "Silesian": "szl_Latn",
        "Tamil": "tam_Taml",
        "Tatar": "tat_Cyrl",
        "Telugu": "tel_Telu",
        "Tajik": "tgk_Cyrl",
        "Tagalog": "tgl_Latn",
        "Thai": "tha_Thai",
        "Tigrinya": "tir_Ethi",
        "Tamasheq (Latin script)": "taq_Latn",
        "Tamasheq (Tifinagh script)": "taq_Tfng",
        "Tok Pisin": "tpi_Latn",
        "Tswana": "tsn_Latn",
        "Tsonga": "tso_Latn",
        "Turkmen": "tuk_Latn",
        "Tumbuka": "tum_Latn",
        "Turkish": "tur_Latn",
        "Twi": "twi_Latn",
        "Central Atlas Tamazight": "tzm_Tfng",
        "Uyghur": "uig_Arab",
        "Ukrainian": "ukr_Cyrl",
        "Umbundu": "umb_Latn",
        "Urdu": "urd_Arab",
        "Northern Uzbek": "uzn_Latn",
        "Venetian": "vec_Latn",
        "Vietnamese": "vie_Latn",
        "Waray": "war_Latn",
        "Wolof": "wol_Latn",
        "Xhosa": "xho_Latn",
        "Eastern Yiddish": "ydd_Hebr",
        "Yoruba": "yor_Latn",
        "Yue Chinese": "yue_Hant",
        "Chinese (Simplified)": "zho_Hans",
        "Chinese (Simplified Han)": "zho_Hans",
        "Chinese (Traditional)": "zho_Hant",
        "Chinese (Traditional Han)": "zho_Hant",
        "Standard Malay": "zsm_Latn",
        "Zulu": "zul_Latn",
    }


def map_language_code(lang_name: str, to_nllb_format: bool = True) -> str:
    """
    Map between common language names and NLLB language codes.

    Args:
        lang_name: Language name or code
        to_nllb_format: If True, convert to NLLB format; if False, convert from NLLB format

    Returns:
        Mapped language code or name
    """
    # Get comprehensive language mapping
    nllb_languages = get_nllb_languages()
    
    # Mapping between common language names and NLLB codes
    # Keys are 2-letter codes, values are NLLB codes (for backward compatibility)
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
    Used for UI components that require integer IDs.
    """
    if not lang_code:
        return 0
    
    # Use a deterministic mapping for known languages
    languages = get_nllb_languages()
    sorted_codes = sorted(languages.values())
    
    try:
        return sorted_codes.index(lang_code) + 1
    except ValueError:
        # Stable fallback for unknown codes
        import hashlib
        return int(hashlib.md5(lang_code.encode()).hexdigest(), 16) % (2**31)


def number_to_language_code(lang_number: int) -> str:
    """
    Convert numeric identifier back to language code.
    """
    if lang_number <= 0:
        return 'eng_Latn'
    
    languages = get_nllb_languages()
    sorted_codes = sorted(languages.values())
    
    if 1 <= lang_number <= len(sorted_codes):
        return sorted_codes[lang_number - 1]
    
    return 'eng_Latn'



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


