"""
Model management service for the Offline Audio Dubbing application.

This module handles scanning, validating, and managing local model files
for Whisper, NLLB, and XTTS.
"""

import os
import logging
import glob
from typing import List, Dict, Optional, Tuple, Union
from src.utils.common.app_config import get_config


class ModelManager:
    """Manages local model files for the application."""

    @staticmethod
    def scan_whisper_models() -> List[Dict[str, str]]:
        """
        Scan for Whisper models (.bin or .gguf).
        
        Returns:
            List of dicts with 'name' and 'path'
        """
        config = get_config()
        whisper_dir = config.WHISPER_MODELS_DIR
        models = []
        
        if not os.path.exists(whisper_dir):
            return models
            
        extensions = ['.bin', '.gguf']
        try:
            for file in os.listdir(whisper_dir):
                if any(file.lower().endswith(ext) for ext in extensions):
                    models.append({
                        "name": file,
                        "path": os.path.join(whisper_dir, file)
                    })
        except OSError as e:
            logging.error(f"Error scanning Whisper models: {e}")
            
        return sorted(models, key=lambda x: x["name"])

    @staticmethod
    def validate_nllb_directory(directory: str) -> bool:
        """
        Validate if a directory contains a valid NLLB model.
        
        Args:
            directory: Path to the directory to check
            
        Returns:
            True if valid, False otherwise
        """
        required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json', 'generation_config.json']
        try:
            found_count = sum(1 for f in required_files if os.path.exists(os.path.join(directory, f)))
            # Either pytorch_model.bin exists or at least 2 required files exist
            if found_count >= 1 and os.path.exists(os.path.join(directory, 'pytorch_model.bin')):
                return True
            return found_count >= 2
        except OSError:
            return False

    @classmethod
    def scan_nllb_models(cls) -> List[Dict[str, str]]:
        """
        Scan for NLLB models.
        
        Returns:
            List of dicts with 'name' and 'path'
        """
        config = get_config()
        nllb_dir = config.NLLB_MODELS_DIR
        models = []
        
        if not os.path.exists(nllb_dir):
            return models
            
        # Check main directory
        if cls.validate_nllb_directory(nllb_dir):
            models.append({"name": "Default NLLB Model", "path": nllb_dir})
            
        # Check subdirectories
        try:
            for item in os.listdir(nllb_dir):
                item_path = os.path.join(nllb_dir, item)
                if os.path.isdir(item_path) and cls.validate_nllb_directory(item_path):
                    models.append({"name": item, "path": item_path})
        except OSError as e:
            logging.error(f"Error scanning NLLB models: {e}")
            
        return sorted(models, key=lambda x: x["name"])

    @staticmethod
    def validate_xtts_directory(directory: str) -> bool:
        """
        Validate if a directory contains a valid XTTS model.
        
        Args:
            directory: Path to the directory to check
            
        Returns:
            True if valid, False otherwise
        """
        required_files = ['config.json', 'model.pth', 'vocab.json', 'speakers.pth', 'language_ids.json']
        try:
            found_count = sum(1 for f in required_files if os.path.exists(os.path.join(directory, f)))
            # Either model.pth exists or at least 2 required files exist
            if found_count >= 1 and os.path.exists(os.path.join(directory, 'model.pth')):
                return True
            return found_count >= 2
        except OSError:
            return False

    @classmethod
    def scan_xtts_models(cls) -> List[Dict[str, str]]:
        """
        Scan for XTTS models.
        
        Returns:
            List of dicts with 'name' and 'path'
        """
        config = get_config()
        xtts_dir = config.XTTS_MODELS_DIR
        models = []
        
        if not os.path.exists(xtts_dir):
            return models
            
        # Check main directory
        if cls.validate_xtts_directory(xtts_dir):
            models.append({"name": "Default XTTS Model", "path": xtts_dir})
            
        # Check subdirectories
        try:
            for item in os.listdir(xtts_dir):
                item_path = os.path.join(xtts_dir, item)
                if os.path.isdir(item_path) and cls.validate_xtts_directory(item_path):
                    models.append({"name": item, "path": item_path})
        except OSError as e:
            logging.error(f"Error scanning XTTS models: {e}")
            
        return sorted(models, key=lambda x: x["name"])

    @classmethod
    def all_models_available(cls) -> bool:
        """Check if at least one of each model type is available."""
        return (bool(cls.scan_whisper_models()) and 
                bool(cls.scan_nllb_models()) and 
                bool(cls.scan_xtts_models()))
