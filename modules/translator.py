from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import logging
import os


class Translator:
    def __init__(self, model_path):
        """
        Initialize the NLLB translator with the specified model.

        Args:
            model_path (str): Path to the NLLB model directory
        """
        # Validate model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"NLLB model not found at {model_path}")
        
        try:
            # Load tokenizer and model
            logging.info(f"Loading NLLB tokenizer from {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            logging.info(f"Loading NLLB model from {model_path}")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

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
        if src_lang not in self.tokenizer.lang_code_to_id:
            raise ValueError(f"Source language '{src_lang}' not supported by model")
        
        if tgt_lang not in self.tokenizer.lang_code_to_id:
            raise ValueError(f"Target language '{tgt_lang}' not supported by model")
        
        try:
            logging.info(f"Starting translation from {src_lang} to {tgt_lang}")
            
            # Tokenize the input text
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)

            # Generate translation
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=min(len(text) * 2, 512),  # Adaptive max length based on input
                    num_beams=5,
                    early_stopping=True,
                    decoder_start_token_id=self.tokenizer.lang_code_to_id[tgt_lang],
                    no_repeat_ngram_size=3,
                    temperature=0.7,
                    top_p=0.9
                )

            # Decode the translated text
            translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            logging.info(f"Translation completed successfully")
            return translated_text
            
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