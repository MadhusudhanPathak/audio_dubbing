"""
Dialog windows for the Offline Audio Dubbing application.

This module contains reusable dialog components for the GUI interface,
including model information and status dialogs.
"""

import os
import logging
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QGridLayout, QLabel, QCheckBox, QPushButton
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class ModelInfoDialog(QDialog):
    """Modal dialog showing required model downloads with local availability check."""

    def __init__(self, parent=None):
        """
        Initialize the model information dialog.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Required Model Downloads")
        self.setModal(True)
        self.setGeometry(200, 200, 800, 500)
        self.init_ui()

    def init_ui(self):
        """Initialize the dialog UI components."""
        layout = QVBoxLayout()
        grid_layout = QGridLayout()

        # Whisper model section
        whisper_label = QLabel("<b>Whisper:</b>")
        whisper_label.setTextFormat(Qt.RichText)
        grid_layout.addWidget(whisper_label, 0, 0)

        whisper_info = QLabel(
            "<a href='https://huggingface.co/ggerganov/whisper.cpp/tree/main'>"
            "https://huggingface.co/ggerganov/whisper.cpp/tree/main</a><br>"
            "Expected extension: <b>.bin</b> (ggml format) or <b>.gguf</b> (GGUF format)<br>"
            "Place in: <b>Models/whisper/</b><br>"
            "Common models: ggml-tiny.bin, ggml-base.bin, ggml-small.bin, ggml-medium.bin, ggml-large.bin"
        )
        whisper_info.setTextFormat(Qt.RichText)
        whisper_info.setOpenExternalLinks(True)
        whisper_info.setWordWrap(True)
        grid_layout.addWidget(whisper_info, 0, 1)

        whisper_available = self.check_whisper_models()
        self.whisper_checkbox = QCheckBox("Available locally")
        self.whisper_checkbox.setChecked(whisper_available)
        grid_layout.addWidget(self.whisper_checkbox, 0, 2)

        # NLLB model section
        nllb_label = QLabel("<b>NLLB:</b>")
        nllb_label.setTextFormat(Qt.RichText)
        grid_layout.addWidget(nllb_label, 1, 0)

        nllb_info = QLabel(
            "<a href='https://huggingface.co/facebook/nllb-200-distilled-600M'>"
            "https://huggingface.co/facebook/nllb-200-distilled-600M</a><br>"
            "Expected: <b>model directories</b> containing config.json, pytorch_model.bin, "
            "tokenizer.json, generation_config.json<br>"
            "Place in: <b>Models/nllb/</b>"
        )
        nllb_info.setTextFormat(Qt.RichText)
        nllb_info.setOpenExternalLinks(True)
        nllb_info.setWordWrap(True)
        grid_layout.addWidget(nllb_info, 1, 1)

        nllb_available = self.check_nllb_models()
        self.nllb_checkbox = QCheckBox("Available locally")
        self.nllb_checkbox.setChecked(nllb_available)
        grid_layout.addWidget(self.nllb_checkbox, 1, 2)

        # XTTS-v2 model section
        xtts_label = QLabel("<b>XTTS-v2:</b>")
        xtts_label.setTextFormat(Qt.RichText)
        grid_layout.addWidget(xtts_label, 2, 0)

        xtts_info = QLabel(
            "<a href='https://huggingface.co/coqui/XTTS-v2'>"
            "https://huggingface.co/coqui/XTTS-v2</a><br>"
            "Expected: <b>model directories</b> containing config.json, model.pth, vocab.json, "
            "speakers.pth, language_ids.json<br>"
            "Place in: <b>Models/xtts/</b>"
        )
        xtts_info.setTextFormat(Qt.RichText)
        xtts_info.setOpenExternalLinks(True)
        xtts_info.setWordWrap(True)
        grid_layout.addWidget(xtts_info, 2, 1)

        xtts_available = self.check_xtts_models()
        self.xtts_checkbox = QCheckBox("Available locally")
        self.xtts_checkbox.setChecked(xtts_available)
        grid_layout.addWidget(self.xtts_checkbox, 2, 2)

        layout.addLayout(grid_layout)

        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)

        self.setLayout(layout)

    @staticmethod
    def check_whisper_models() -> bool:
        """
        Check if Whisper models are available locally.

        Returns:
            True if models are found, False otherwise
        """
        whisper_dir = "./Models/whisper"
        if not os.path.exists(whisper_dir):
            return False

        whisper_extensions = ['.bin', '.gguf']
        try:
            for file in os.listdir(whisper_dir):
                if any(file.lower().endswith(ext) for ext in whisper_extensions):
                    return True
        except OSError as e:
            logging.warning(f"Error checking whisper models: {e}")
        return False

    @staticmethod
    def check_nllb_models() -> bool:
        """
        Check if NLLB models are available locally.

        Returns:
            True if models are found, False otherwise
        """
        nllb_dir = "./Models/nllb"
        if not os.path.exists(nllb_dir):
            return False

        nllb_required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json', 'generation_config.json']

        try:
            # Check for required files directly in the directory
            direct_files_count = sum(
                1 for file in nllb_required_files
                if os.path.exists(os.path.join(nllb_dir, file))
            )
            if direct_files_count >= 1 and os.path.exists(os.path.join(nllb_dir, 'pytorch_model.bin')):
                return True
            elif direct_files_count >= 2:
                return True

            # Check for model files in subdirectories
            for item in os.listdir(nllb_dir):
                item_path = os.path.join(nllb_dir, item)
                if not os.path.isdir(item_path):
                    continue

                model_files_count = sum(
                    1 for file in nllb_required_files
                    if os.path.exists(os.path.join(item_path, file))
                )
                if model_files_count >= 1 and os.path.exists(os.path.join(item_path, 'pytorch_model.bin')):
                    return True
                elif model_files_count >= 2:
                    return True
        except OSError as e:
            logging.warning(f"Error checking NLLB models: {e}")
        return False

    @staticmethod
    def check_xtts_models() -> bool:
        """
        Check if XTTS models are available locally.

        Returns:
            True if models are found, False otherwise
        """
        xtts_dir = "./Models/xtts"
        if not os.path.exists(xtts_dir):
            return False

        xtts_required_files = ['config.json', 'model.pth', 'vocab.json', 'speakers.pth', 'language_ids.json']

        try:
            # Check for required files directly in the directory
            direct_files_count = sum(
                1 for file in xtts_required_files
                if os.path.exists(os.path.join(xtts_dir, file))
            )
            if direct_files_count >= 1 and os.path.exists(os.path.join(xtts_dir, 'model.pth')):
                return True
            elif direct_files_count >= 2:
                return True

            # Check for model files in subdirectories
            for item in os.listdir(xtts_dir):
                item_path = os.path.join(xtts_dir, item)
                if not os.path.isdir(item_path):
                    continue

                model_files_count = sum(
                    1 for file in xtts_required_files
                    if os.path.exists(os.path.join(item_path, file))
                )
                if model_files_count >= 1 and os.path.exists(os.path.join(item_path, 'model.pth')):
                    return True
                elif model_files_count >= 2:
                    return True
        except OSError as e:
            logging.warning(f"Error checking XTTS models: {e}")
        return False

    def refresh_model_status(self):
        """Refresh the status of all model availability checks."""
        self.whisper_checkbox.setChecked(self.check_whisper_models())
        self.nllb_checkbox.setChecked(self.check_nllb_models())
        self.xtts_checkbox.setChecked(self.check_xtts_models())
