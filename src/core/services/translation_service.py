import torch
import logging
import os
from typing import Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig


class TranslationError(Exception):
    """Custom exception for translation-related errors."""
    pass


class Translator:
    """NLLB-based text translation service."""

    def __init__(self, model_path: str, use_quantization: bool = False, device: Optional[str] = None):
        self.model_path = model_path
        self.use_quantization = use_quantization

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"NLLB model not found at {model_path}")

        try:
            self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)

            model_kwargs = {
                'local_files_only': True,
                'torch_dtype': torch.float16 if self.device.type == 'cuda' else torch.float32,
            }

            if self.use_quantization and self.device.type == 'cuda':
                model_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)

            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, **model_kwargs).to(self.device)
            self.model.eval()

            self.lang_code_to_id = getattr(self.tokenizer, 'lang_code_to_id', {})
        except Exception as e:
            raise TranslationError(f"Failed to load NLLB model: {str(e)}")

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """
        Translate text.
        
        Args:
            text: Text to translate
            src_lang: Source NLLB code
            tgt_lang: Target NLLB code
            
        Returns:
            Translated text
        """
        if not text.strip(): return ""

        try:
            if src_lang and hasattr(self.tokenizer, "src_lang"):
                self.tokenizer.src_lang = src_lang

            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.lang_code_to_id.get(tgt_lang),
                    max_length=512,
                    num_beams=5
                )

            return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        except Exception as e:
            raise TranslationError(f"Translation failed: {str(e)}")

    def unload(self) -> None:
        """Free GPU memory after use. Safe to call more than once."""
        if getattr(self, "model", None) is not None:
            del self.model
            self.model = None
        if getattr(self, "tokenizer", None) is not None:
            del self.tokenizer
            self.tokenizer = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        logging.info("Translator unloaded.")