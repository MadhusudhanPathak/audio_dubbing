"""
Model management service for the Offline Audio Dubbing application.

This module handles scanning, validating, and managing local model files
for MOSS-Audio, NLLB, and MOSS-TTS.
"""

import os
import logging
from typing import List, Dict
from src.utils.common.app_config import get_config
from src.utils.model_setup_checker import run_setup_check, REQUIRED_DIRS


class ModelManager:
    """Manages local model files for the application."""

    @staticmethod
    def _scan_tier_model(dir_key: str, label: str, tier_label: str) -> List[Dict[str, str]]:
        """Scan a single setup-checker-managed model directory for a config.json."""
        model_path = str(REQUIRED_DIRS[dir_key])
        if os.path.exists(os.path.join(model_path, "config.json")):
            return [{"name": f"{label} ({tier_label})", "path": model_path}]
        return []

    @staticmethod
    def scan_moss_audio_models() -> List[Dict[str, str]]:
        """
        Scan for MOSS-Audio (STT) models using the setup checker.

        Returns:
            List of dicts with 'name' and 'path'
        """
        try:
            _, tier = run_setup_check(show_gui_dialog=False)
            return ModelManager._scan_tier_model(tier.stt_dir, "MOSS-Audio", tier.label)
        except Exception as e:
            logging.error(f"Error scanning MOSS-Audio models: {e}")
            return []

    @staticmethod
    def scan_nllb_models() -> List[Dict[str, str]]:
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

        if ModelManager.validate_nllb_directory(nllb_dir):
            models.append({"name": "Default NLLB Model", "path": nllb_dir})

        try:
            for item in os.listdir(nllb_dir):
                item_path = os.path.join(nllb_dir, item)
                if os.path.isdir(item_path) and ModelManager.validate_nllb_directory(item_path):
                    models.append({"name": item, "path": item_path})
        except OSError as e:
            logging.error(f"Error scanning NLLB models: {e}")

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
            return all(os.path.exists(os.path.join(directory, f)) for f in required_files)
        except OSError:
            return False

    @staticmethod
    def scan_moss_tts_models() -> List[Dict[str, str]]:
        """
        Scan for MOSS-TTS (voice synthesis) models using the setup checker.

        Returns:
            List of dicts with 'name' and 'path'
        """
        try:
            _, tier = run_setup_check(show_gui_dialog=False)
            return ModelManager._scan_tier_model(tier.tts_dir, "MOSS-TTS", tier.label)
        except Exception as e:
            logging.error(f"Error scanning MOSS-TTS models: {e}")
            return []

    @classmethod
    def all_models_available(cls) -> bool:
        """Check if at least one of each model type is available."""
        return (bool(cls.scan_moss_audio_models()) and
                bool(cls.scan_nllb_models()) and
                bool(cls.scan_moss_tts_models()))
