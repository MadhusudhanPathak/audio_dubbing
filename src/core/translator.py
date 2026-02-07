from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import logging
import os
from src.config.app_config import get_config


class Translator:
    def __init__(self, model_path):
        """
        Initialize the NLLB translator with the specified model.

        Args:
            model_path (str): Path to the NLLB model directory
        
        Raises:
            FileNotFoundError: If model path or required files don't exist
            ValueError: If model_path is invalid
            RuntimeError: If model loading fails
        """
        # Validate inputs
        if not model_path or not isinstance(model_path, str):
            raise ValueError("Model path must be a non-empty string")
        
        # Validate model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"NLLB model not found at {model_path}")

        # Check if required model files exist
        # First check for safetensors format (more secure)
        safetensors_files = ['config.json', 'model.safetensors']
        traditional_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
        
        # Check if safetensors format exists
        has_safetensors = all(os.path.exists(os.path.join(model_path, f)) for f in ['config.json', 'model.safetensors'])
        
        if has_safetensors:
            logging.info("Detected safetensors format model files")
            # Safetensors format doesn't need pytorch_model.bin
        else:
            # Check traditional format
            missing_files = []
            for file in traditional_files:
                if not os.path.exists(os.path.join(model_path, file)):
                    missing_files.append(file)
            
            if missing_files:
                raise FileNotFoundError(f"NLLB model is missing required files: {missing_files}")

        try:
            # Load tokenizer and model
            logging.info(f"Loading NLLB tokenizer from {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True  # Only use local files, don't download
            )

            logging.info(f"Loading NLLB model from {model_path}")
            
            # If safetensors format exists, force use of safetensors
            if has_safetensors:
                logging.info("Loading model using safetensors format (secure)")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_path,
                    local_files_only=True,  # Only use local files, don't download
                    torch_dtype=torch.float32,  # Use float32 to reduce memory usage
                    trust_remote_code=False,  # Explicitly disable remote code execution
                    use_safetensors=True  # Force safetensors format
                )
            else:
                # For traditional format, try safetensors first (might work if other safetensors files exist)
                try:
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_path,
                        local_files_only=True,  # Only use local files, don't download
                        torch_dtype=torch.float32,  # Use float32 to reduce memory usage
                        trust_remote_code=False,  # Explicitly disable remote code execution
                        use_safetensors=True  # Try safetensors format
                    )
                    logging.info("Successfully loaded model using safetensors format")
                except Exception as safetensors_error:
                    logging.warning(f"Safetensors loading failed: {safetensors_error}. Falling back to traditional format...")
                    # If safetensors fails, try traditional format
                    # Note: This will still trigger the vulnerability warning if PyTorch < 2.6
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_path,
                        local_files_only=True,  # Only use local files, don't download
                        torch_dtype=torch.float32,  # Use float32 to reduce memory usage
                        trust_remote_code=False,  # Explicitly disable remote code execution
                        use_safetensors=False  # Use traditional format
                    )

            # Determine device (GPU if available, otherwise CPU)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logging.info(f"Using device: {self.device}")

            self.model.to(self.device)
            logging.info("NLLB model loaded successfully")

        except Exception as e:
            logging.error(f"Failed to load NLLB model: {str(e)}")
            raise

    def translate(self, text, src_lang, tgt_lang):
        """
        Translate text from source language to target language using NLLB model.

        Args:
            text (str): Text to translate
            src_lang (str): Source language code (e.g., 'eng_Latn', 'spa_Latn')
            tgt_lang (str): Target language code (e.g., 'eng_Latn', 'spa_Latn')

        Returns:
            str: Translated text
        """
        # Validate inputs
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")

        if not src_lang or not tgt_lang:
            raise ValueError("Source and target languages must be specified")

        # Validate language codes exist in tokenizer
        # Different versions of transformers may have different attributes
        if hasattr(self.tokenizer, 'lang_code_to_id'):
            lang_code_mapping = self.tokenizer.lang_code_to_id
        elif hasattr(self.tokenizer, 'language_code_to_id'):
            lang_code_mapping = self.tokenizer.language_code_to_id
        else:
            # Try to get language codes from the tokenizer config
            try:
                # Access the language codes differently based on tokenizer type
                if hasattr(self.tokenizer, 'config') and 'lang_codes' in self.tokenizer.config:
                    lang_codes = self.tokenizer.config['lang_codes']
                    lang_code_mapping = {lang: idx for idx, lang in enumerate(lang_codes)}
                else:
                    # Fallback: try to get from model config
                    if hasattr(self.model.config, 'lang_code_to_id'):
                        lang_code_mapping = self.model.config.lang_code_to_id
                    else:
                        # Last resort: try to infer from tokenizer's special tokens
                        lang_code_mapping = {}
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
                            'war_Latn', 'wol_Latn', 'xho_Latn', 'ydd_Hebr', 'yor_Latn', 'yue_Hant', 'zho_Hans', 'zho_Hant', 'zsm_Latn', 'zul_Latn'
                        ]
                        lang_code_mapping = {lang: idx for idx, lang in enumerate(common_lang_codes)}
            except Exception:
                raise ValueError(f"Unable to determine language codes for this tokenizer. Language '{tgt_lang}' not supported.")

        if src_lang not in lang_code_mapping:
            available_langs = list(lang_code_mapping.keys())[:10]  # Show first 10 languages
            raise ValueError(f"Source language '{src_lang}' not supported by model. Available languages: {available_langs}...")

        if tgt_lang not in lang_code_mapping:
            available_langs = list(lang_code_mapping.keys())[:10]  # Show first 10 languages
            raise ValueError(f"Target language '{tgt_lang}' not supported by model. Available languages: {available_langs}...")

        try:
            logging.info(f"Starting translation from {src_lang} to {tgt_lang}")
            
            # Handle empty text case
            if not text.strip():
                logging.warning("Empty text provided for translation, returning empty string")
                return ""

            # For NLLB models, we need to tokenize with language codes
            # The correct way is to use the tokenizer with src_lang and tgt_lang parameters
            # But if that fails, we'll use the language ID mapping in the generation step
            try:
                # Try the standard approach first
                inputs = self.tokenizer(
                    text,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=512
                ).to(self.device)
                
                # If this works, proceed with generation
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=min(len(text) * 2, 512),
                        num_beams=5,
                        early_stopping=True,
                        no_repeat_ngram_size=3,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True
                    )
            except TypeError:
                # If the tokenizer doesn't support src_lang/tgt_lang, use the language ID mapping approach
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=512
                ).to(self.device)

                # Generate translation with proper language ID handling for NLLB
                try:
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_length=min(len(text) * 2, 512),  # Adaptive max length based on input
                            num_beams=5,
                            early_stopping=True,
                            decoder_start_token_id=self.tokenizer.convert_tokens_to_ids(tgt_lang),  # Use the token ID for target language
                            no_repeat_ngram_size=3,
                            temperature=0.7,
                            top_p=0.9,
                            do_sample=True  # Enable sampling for better translations
                        )
                except:
                    # If convert_tokens_to_ids fails, use the lang_code_mapping as fallback
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_length=min(len(text) * 2, 512),  # Adaptive max length based on input
                            num_beams=5,
                            early_stopping=True,
                            decoder_start_token_id=lang_code_mapping[tgt_lang],  # Use the target language ID for decoder start
                            no_repeat_ngram_size=3,
                            temperature=0.7,
                            top_p=0.9,
                            do_sample=True  # Enable sampling for better translations
                        )

            # Decode the translated text
            translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            logging.info(f"Translation completed successfully")
            return translated_text

        except torch.cuda.OutOfMemoryError:
            logging.error("CUDA out of memory during translation. Try using a smaller input text or a less resource-intensive model.")
            raise RuntimeError("CUDA out of memory during translation. Try using a smaller input text.")
        except Exception as e:
            logging.error(f"Error during translation: {str(e)}")
            raise

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
        except:
            pass  # Ignore errors during cleanup


# Example usage:
if __name__ == "__main__":
    # Example of how to use the Translator class
    # translator = Translator("path/to/nllb/model/")
    # result = translator.translate("Hello world", "eng_Latn", "spa_Latn")
    # print(result)
    pass