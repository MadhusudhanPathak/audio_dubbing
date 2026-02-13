"""
Enhanced NLLB Translation Module for Offline Audio Dubbing

This module provides advanced translation capabilities using NLLB (No Language Left Behind) models.
It handles text translation between multiple languages with improved error handling, memory management,
and performance optimizations.

The module supports:
- Multiple NLLB model variants (distilled and full versions)
- Memory-efficient quantization options
- Batch translation capabilities
- Comprehensive error handling and logging
- Automatic device detection (CPU/GPU)
- Extensive language support (200+ languages)

Key features:
- Quantization support for reduced memory usage
- Proper cleanup of GPU memory
- Validation of language codes
- Batch processing for multiple texts
- Character-based language detection heuristic
"""

import torch
import logging
import os
from typing import Optional, Dict, Any, List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from src.utils.common.app_config import get_config
from src.utils.common.helpers import language_code_to_number, number_to_language_code


class TranslationError(Exception):
    """Custom exception for translation-related errors.

    Raised when translation operations fail due to model issues,
    memory constraints, or other translation-specific problems.
    """
    pass


class ModelLoadError(Exception):
    """Custom exception for model loading errors.

    Raised when the NLLB model fails to load due to missing files,
    incompatible formats, or other model loading issues.
    """
    pass


class InvalidLanguageError(Exception):
    """Custom exception for invalid language codes.

    Raised when unsupported or incorrectly formatted language codes
    are provided to translation functions.
    """
    pass


class Translator:
    """
    Enhanced NLLB Translator with improved performance and error handling.

    This class handles text translation between multiple languages using NLLB models.
    It includes memory optimization, proper error handling, and support for various
    model formats.
    """

    def __init__(self, model_path: str, use_quantization: bool = False, device: Optional[str] = None):
        """
        Initialize the NLLB translator with the specified model.

        Args:
            model_path (str): Path to the NLLB model directory
            use_quantization (bool): Whether to use 8-bit quantization for memory efficiency
            device (str, optional): Device to run the model on ('cuda' or 'cpu').
                                   If None, automatically detects available device.

        Raises:
            ModelLoadError: If model loading fails
            FileNotFoundError: If model path or required files don't exist
            InvalidLanguageError: If language validation fails
        """
        self.model_path = model_path
        self.use_quantization = use_quantization

        # Validate inputs
        if not model_path or not isinstance(model_path, str):
            raise ValueError("Model path must be a non-empty string")

        # Validate model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"NLLB model not found at {model_path}")

        # Check if required model files exist
        required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json', 'generation_config.json']

        missing_files = []
        for file in required_files:
            if not os.path.exists(os.path.join(model_path, file)):
                missing_files.append(file)

        if missing_files:
            logging.warning(f"NLLB model is missing some required files: {missing_files}")
            # Check for alternative model file formats
            if not os.path.exists(os.path.join(model_path, 'model.safetensors')) and \
               not os.path.exists(os.path.join(model_path, 'pytorch_model.bin')):
                raise FileNotFoundError(f"NLLB model is missing required files: {missing_files}")

        try:
            # Determine device (GPU if available, otherwise CPU)
            if device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)

            logging.info(f"Using device: {self.device}")

            # Load tokenizer
            logging.info(f"Loading NLLB tokenizer from {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,  # Only use local files, don't download
                use_fast=True  # Use fast tokenizer if available
            )

            # Configure model loading based on quantization preference
            model_kwargs = {
                'local_files_only': True,
                'torch_dtype': torch.float16 if self.device.type == 'cuda' else torch.float32,
                'trust_remote_code': False,
            }

            if self.use_quantization and self.device.type == 'cuda':
                # Use 8-bit quantization for memory efficiency
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                )
                model_kwargs['quantization_config'] = bnb_config
                logging.info("Using 8-bit quantization for memory efficiency")
            else:
                # Use float16 for CUDA or float32 for CPU
                model_kwargs['torch_dtype'] = torch.float16 if self.device.type == 'cuda' else torch.float32

            # Load the model
            logging.info(f"Loading NLLB model from {model_path}")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                **model_kwargs
            )

            # Move model to device
            self.model.to(self.device)

            # Set model to evaluation mode
            self.model.eval()

            logging.info("NLLB model loaded successfully")

            # Cache language codes for faster lookup
            self._cache_language_codes()

        except Exception as e:
            logging.error(f"Failed to load NLLB model: {str(e)}")
            raise ModelLoadError(f"Failed to load NLLB model: {str(e)}")

    def _cache_language_codes(self):
        """Cache language codes for faster lookup during translation."""
        try:
            # Try to get language codes from tokenizer
            if hasattr(self.tokenizer, 'lang_code_to_id'):
                self.lang_code_to_id = self.tokenizer.lang_code_to_id
            elif hasattr(self.model.config, 'lang_code_to_id'):
                self.lang_code_to_id = self.model.config.lang_code_to_id
            else:
                # Fallback to common NLLB language codes
                self.lang_code_to_id = self._get_common_language_codes()

            logging.info(f"Cached {len(self.lang_code_to_id)} language codes")
            logging.debug(f"Available language codes: {list(self.lang_code_to_id.keys())}")
            
            # Log specific language IDs for debugging
            for lang_code in ['eng_Latn', 'ita_Latn', 'ara_Arab', 'spa_Latn', 'hin_Deva']:
                if lang_code in self.lang_code_to_id:
                    logging.debug(f"Language '{lang_code}' has ID: {self.lang_code_to_id[lang_code]}")
                    
        except Exception as e:
            logging.warning(f"Could not cache language codes: {e}")
            self.lang_code_to_id = self._get_common_language_codes()

    def _get_common_language_codes(self) -> Dict[str, int]:
        """
        Get a dictionary of common NLLB language codes.

        Returns:
            Dict[str, int]: Mapping of language codes to IDs
        """
        # Common NLLB language codes
        common_lang_codes = [
            'ace_Arab', 'ace_Latn', 'ady_Cyrl', 'aeb_Arab', 'afr_Latn', 'ajp_Arab', 'aka_Latn',
            'amh_Ethi', 'apc_Arab', 'arb_Arab', 'ars_Arab', 'ary_Arab', 'arz_Arab', 'asm_Beng',
            'ast_Latn', 'awa_Deva', 'ayr_Latn', 'azb_Arab', 'aze_Latn', 'bak_Cyrl', 'bam_Latn',
            'ban_Latn', 'bel_Cyrl', 'bem_Latn', 'ben_Beng', 'bho_Deva', 'bjn_Arab', 'bjn_Latn',
            'bod_Tibt', 'bos_Latn', 'bug_Latn', 'bul_Cyrl', 'cat_Latn', 'ceb_Latn', 'ces_Latn',
            'cjk_Latn', 'ckb_Arab', 'crh_Latn', 'cym_Latn', 'dan_Latn', 'deu_Latn', 'dik_Latn',
            'dyu_Latn', 'dzo_Tibt', 'ell_Grek', 'eng_Latn', 'epo_Latn', 'est_Latn', 'eus_Latn',
            'ewe_Latn', 'fao_Latn', 'fij_Latn', 'fin_Latn', 'fon_Latn', 'fra_Latn', 'fur_Latn',
            'fuv_Latn', 'gla_Latn', 'gle_Latn', 'glg_Latn', 'grn_Latn', 'guj_Gujr', 'hat_Latn',
            'hau_Latn', 'heb_Hebr', 'hin_Deva', 'hne_Deva', 'hrv_Latn', 'hun_Latn', 'hye_Armn',
            'ibo_Latn', 'ilo_Latn', 'ind_Latn', 'isl_Latn', 'ita_Latn', 'jav_Latn', 'jpn_Jpan',
            'kab_Latn', 'kac_Latn', 'kam_Latn', 'kan_Knda', 'kas_Arab', 'kas_Deva', 'kat_Geor',
            'kaz_Cyrl', 'kbp_Latn', 'kea_Latn', 'khm_Khmr', 'kik_Latn', 'kin_Latn', 'kir_Cyrl',
            'kmb_Latn', 'kmr_Latn', 'knc_Arab', 'knc_Latn', 'kon_Latn', 'kor_Hang', 'lao_Laoo',
            'lij_Latn', 'lim_Latn', 'lin_Latn', 'lit_Latn', 'lmo_Latn', 'ltg_Latn', 'ltz_Latn',
            'lua_Latn', 'lug_Latn', 'luo_Latn', 'lus_Latn', 'lvs_Latn', 'mag_Deva', 'mai_Deva',
            'mal_Mlym', 'mar_Deva', 'min_Arab', 'min_Latn', 'mkd_Cyrl', 'mlt_Latn', 'mni_Beng',
            'mos_Latn', 'mri_Latn', 'mya_Mymr', 'nld_Latn', 'nno_Latn', 'nob_Latn', 'npi_Deva',
            'nso_Latn', 'nus_Latn', 'nya_Latn', 'oci_Latn', 'ory_Orya', 'pag_Latn', 'pan_Guru',
            'pap_Latn', 'pol_Latn', 'por_Latn', 'prs_Arab', 'pus_Arab', 'ron_Latn', 'run_Latn',
            'rus_Cyrl', 'sag_Latn', 'san_Deva', 'sat_Olck', 'scn_Latn', 'shn_Mymr', 'sin_Sinh',
            'slk_Latn', 'slv_Latn', 'smo_Latn', 'sna_Latn', 'snd_Arab', 'som_Latn', 'sot_Latn',
            'spa_Latn', 'sqi_Latn', 'srd_Latn', 'srp_Cyrl', 'srp_Latn', 'ssw_Latn', 'sun_Latn',
            'swe_Latn', 'swh_Latn', 'szl_Latn', 'tam_Taml', 'taq_Latn', 'taq_Tfng', 'tat_Cyrl',
            'tel_Telu', 'tgk_Cyrl', 'tgl_Latn', 'tha_Thai', 'tur_Latn', 'twi_Latn', 'tzm_Tfng',
            'uig_Arab', 'ukr_Cyrl', 'umb_Latn', 'urd_Arab', 'uzb_Latn', 'vec_Latn', 'vie_Latn',
            'war_Latn', 'wol_Latn', 'xho_Latn', 'ydd_Hebr', 'yor_Latn', 'yue_Hant', 'zho_Hans',
            'zho_Hant', 'zsm_Latn', 'zul_Latn'
        ]

        return {lang: idx for idx, lang in enumerate(common_lang_codes)}

    def _validate_language_codes(self, src_lang: str, tgt_lang: str) -> None:
        """
        Validate that source and target language codes are supported by the model.

        Args:
            src_lang (str): Source language code
            tgt_lang (str): Target language code

        Raises:
            InvalidLanguageError: If language codes are not supported
        """
        logging.debug(f"Validating language codes - src: {src_lang}, tgt: {tgt_lang}")
        logging.debug(f"Available language codes: {list(self.lang_code_to_id.keys())[:10]}...")

        if src_lang not in self.lang_code_to_id:
            available_langs = list(self.lang_code_to_id.keys())[:10]  # Show first 10 languages
            raise InvalidLanguageError(
                f"Source language '{src_lang}' not supported by model. "
                f"Available languages: {available_langs}... "
                f"Full list contains {len(self.lang_code_to_id)} languages."
            )

        if tgt_lang not in self.lang_code_to_id:
            available_langs = list(self.lang_code_to_id.keys())[:10]  # Show first 10 languages
            raise InvalidLanguageError(
                f"Target language '{tgt_lang}' not supported by model. "
                f"Available languages: {available_langs}... "
                f"Full list contains {len(self.lang_code_to_id)} languages."
            )

    def _validate_input_text(self, text: str) -> None:
        """
        Validate input text for translation.

        Args:
            text (str): Text to validate

        Raises:
            ValueError: If text is invalid
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")

        if len(text.strip()) == 0:
            raise ValueError("Text cannot be empty or contain only whitespace")

        # Check for extremely long texts that might cause memory issues
        if len(text) > 10000:  # 10k characters is quite long for a single translation
            logging.warning(f"Input text is very long ({len(text)} characters), "
                           "consider splitting it into smaller chunks for better performance")

    def translate(self, text: str, src_lang: str, tgt_lang: str,
                  max_length: int = 512, num_beams: int = 5) -> str:
        """
        Translate text from source language to target language using NLLB model.

        Args:
            text (str): Text to translate
            src_lang (str): Source language code (e.g., 'eng_Latn', 'spa_Latn')
            tgt_lang (str): Target language code (e.g., 'eng_Latn', 'spa_Latn')
            max_length (int): Maximum length of generated text (default: 512)
            num_beams (int): Number of beams for beam search (default: 5)

        Returns:
            str: Translated text

        Raises:
            TranslationError: If translation fails
            InvalidLanguageError: If language codes are invalid
            ValueError: If input text is invalid
        """
        # Log the incoming language codes for debugging
        logging.info(f"Starting translation from {src_lang} to {tgt_lang}")
        logging.debug(f"Input text: {text[:100]}..." if len(text) > 100 else f"Input text: {text}")
        
        # Validate inputs
        self._validate_input_text(text)
        self._validate_language_codes(src_lang, tgt_lang)

        try:
            logging.debug(f"Input text length: {len(text)} characters")

            # Handle empty text case
            if not text.strip():
                logging.warning("Empty text provided for translation, returning empty string")
                return ""

            # Get target language ID
            tgt_lang_id = self.lang_code_to_id[tgt_lang]
            logging.debug(f"Target language ID for '{tgt_lang}': {tgt_lang_id}")

            # Verify that the source language is also valid
            src_lang_id = self.lang_code_to_id[src_lang]
            logging.debug(f"Source language ID for '{src_lang}': {src_lang_id}")

            # For NLLB models, we need to include the target language in the input
            # The format is usually: [SRC_LANG_CODE] text [TGT_LANG_CODE]
            # Using the special language tokens from the tokenizer
            full_input_text = f"{text}"
            
            # Tokenize with the input text
            inputs = self.tokenizer(
                full_input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # For NLLB models, we need to make sure the target language is properly specified
            # Some implementations require using prepare_seq2seq_batch
            if hasattr(self.tokenizer, 'prepare_seq2seq_batch'):
                # Use the tokenizer's specific method for NLLB
                inputs = self.tokenizer.prepare_seq2seq_batch(
                    [text],
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)

            # Log tokenizer input for debugging
            logging.debug(f"Tokenized input shape: {inputs['input_ids'].shape}")
            
            # Generate translation
            # Since we're using prepare_seq2seq_batch, the target language is already encoded in the input
            # So we don't need to specify decoder_start_token_id separately
            with torch.no_grad():  # Disable gradient computation for inference
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    length_penalty=1.0,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    do_sample=False,  # Deterministic output for consistency
                    temperature=1.0,  # Default temperature
                    top_k=50,        # Limit to top 50 tokens
                    top_p=0.95       # Nucleus sampling
                )

            # Decode the translated text
            translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            logging.info(f"Translation completed successfully from {src_lang} to {tgt_lang}")
            logging.debug(f"Output text length: {len(translated_text)} characters")
            logging.debug(f"Output text: {translated_text[:100]}..." if len(translated_text) > 100 else f"Output text: {translated_text}")

            return translated_text.strip()

        except torch.cuda.OutOfMemoryError:
            logging.error("CUDA out of memory during translation. "
                         "Try using a smaller input text or enable quantization.")
            raise TranslationError(
                "CUDA out of memory during translation. "
                "Try using a smaller input text or enable quantization."
            )
        except Exception as e:
            logging.error(f"Error during translation: {str(e)}")
            raise TranslationError(f"Translation failed: {str(e)}")

    def translate_batch(self, texts: List[str], src_lang: str, tgt_lang: str,
                       max_length: int = 512, num_beams: int = 5) -> List[str]:
        """
        Translate multiple texts from source language to target language.

        This method processes multiple texts in sequence, applying the same
        translation parameters to each. Individual failures are logged but
        don't halt the entire batch processing.

        Args:
            texts (List[str]): List of texts to translate
            src_lang (str): Source language code (e.g., 'eng_Latn', 'spa_Latn')
            tgt_lang (str): Target language code (e.g., 'eng_Latn', 'spa_Latn')
            max_length (int): Maximum length of generated text (default: 512)
            num_beams (int): Number of beams for beam search (default: 5)

        Returns:
            List[str]: List of translated texts, maintaining the same order
                      as the input list. Failed translations return empty strings.

        Raises:
            ValueError: If texts is not a list or is empty
            InvalidLanguageError: If language codes are invalid
        """
        if not texts or not isinstance(texts, list):
            raise ValueError("Texts must be a non-empty list")

        if len(texts) == 0:
            return []

        results = []
        for i, text in enumerate(texts):
            try:
                translated = self.translate(text, src_lang, tgt_lang, max_length, num_beams)
                results.append(translated)
            except Exception as e:
                logging.error(f"Error translating text {i}: {str(e)}")
                results.append("")  # Add empty string for failed translations

        return results

    def get_supported_languages(self) -> List[str]:
        """
        Get a list of supported language codes.

        Returns:
            List[str]: List of supported language codes in BCP-47 format
                      (e.g., 'eng_Latn', 'spa_Latn', 'rus_Cyrl')
        """
        return list(self.lang_code_to_id.keys())

    def detect_language(self, text: str) -> str:
        """
        Detect the language of the input text using character-based heuristics.

        Note: This is a basic implementation using character pattern recognition.
        For production use with high accuracy requirements, consider using
        a dedicated language detection library like 'langdetect' or 'polyglot'.

        Args:
            text (str): Text to detect language for

        Returns:
            str: Detected language code in BCP-47 format (e.g., 'eng_Latn', 'rus_Cyrl')
                 or 'unknown' if detection fails or text is empty

        Example:
            >>> translator = Translator(model_path)
            >>> lang = translator.detect_language("Hello world")
            >>> print(lang)  # Output: 'eng_Latn'
        """
        # This is a simplified language detection based on character patterns
        # For a more robust solution, integrate with a language detection library
        if not text or len(text.strip()) == 0:
            return 'unknown'

        # Simple heuristic based on common language characteristics
        text_lower = text.lower()

        # Check for common language indicators
        if any(char in text for char in 'абвгдеёжзийклmnопрстуфхцчшщъыьэюя'):
            return 'rus_Cyrl'  # Russian
        elif any(char in text for char in '你好世界'):
            return 'zho_Hans'  # Chinese
        elif any(char in text for char in 'こんにちは'):
            return 'jpn_Jpan'  # Japanese
        elif any(char in text for char in 'مرحبا'):
            return 'ara_Arab'  # Arabic
        elif any(char in text for char in 'हिंदी'):
            return 'hin_Deva'  # Hindi
        elif any(char in text for char in 'αβγδεζηθικλμνξοπρστυφχψω'):
            return 'ell_Grek'  # Greek
        else:
            # Default to English for Latin script
            return 'eng_Latn'

    def __del__(self):
        """
        Cleanup method to free resources when object is destroyed.
        """
        try:
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            logging.warning(f"Error during Translator cleanup: {e}")

    def unload_model(self):
        """
        Explicitly unload the model and tokenizer to free memory.

        This method is useful when working with limited memory resources
        or when you need to switch between different models. It removes
        references to the model and tokenizer objects and clears the GPU cache.

        Usage:
            >>> translator = Translator(model_path)
            >>> # ... perform translations ...
            >>> translator.unload_model()  # Free up memory
        """
        try:
            if hasattr(self, 'model'):
                del self.model
                self.model = None
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
                self.tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            logging.info("Model unloaded successfully")
        except Exception as e:
            logging.error(f"Error unloading model: {e}")


# Example usage and testing
if __name__ == "__main__":
    import sys
    import os

    # Add the project root to the path to allow imports
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)

    # Example of how to use the Translator class
    # translator = Translator("path/to/nllb/model/")
    # result = translator.translate("Hello world", "eng_Latn", "spa_Latn")
    # print(result)

    print("Translator module loaded successfully")