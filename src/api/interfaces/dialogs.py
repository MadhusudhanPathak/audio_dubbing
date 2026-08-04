"""
Dialog windows for the Offline Audio Dubbing application.

This module contains reusable dialog components for the GUI interface,
including model information and status dialogs.
"""

import logging
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QGridLayout, QLabel, QCheckBox, QPushButton
)
from PyQt5.QtCore import Qt

from src.core.services.model_manager import ModelManager


def _scan_available(scan_func, label: str) -> bool:
    """Safely check whether a model scan finds at least one model."""
    try:
        return bool(scan_func())
    except Exception as e:
        logging.error(f"Error scanning {label} models: {e}")
        return False


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

        # MOSS-Audio model section
        moss_audio_label = QLabel("<b>MOSS-Audio (STT):</b>")
        moss_audio_label.setTextFormat(Qt.RichText)
        grid_layout.addWidget(moss_audio_label, 0, 0)

        moss_audio_info = QLabel(
            "<a href='https://huggingface.co/OpenMOSS-Team/MOSS-Audio-4B-Instruct'>"
            "https://huggingface.co/OpenMOSS-Team/MOSS-Audio-4B-Instruct</a><br>"
            "Expected: <b>model directory</b> containing config.json and model files<br>"
            "Place in: <b>Models/moss-audio/</b><br>"
            "Also available: 8B-Instruct variant for higher quality"
        )
        moss_audio_info.setTextFormat(Qt.RichText)
        moss_audio_info.setOpenExternalLinks(True)
        moss_audio_info.setWordWrap(True)
        grid_layout.addWidget(moss_audio_info, 0, 1)

        moss_audio_available = _scan_available(ModelManager.scan_moss_audio_models, "MOSS-Audio")
        self.moss_audio_checkbox = QCheckBox("Available locally")
        self.moss_audio_checkbox.setChecked(moss_audio_available)
        grid_layout.addWidget(self.moss_audio_checkbox, 0, 2)

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

        nllb_available = _scan_available(ModelManager.scan_nllb_models, "NLLB")
        self.nllb_checkbox = QCheckBox("Available locally")
        self.nllb_checkbox.setChecked(nllb_available)
        grid_layout.addWidget(self.nllb_checkbox, 1, 2)

        # MOSS-TTS model section
        moss_tts_label = QLabel("<b>MOSS-TTS (Voice Synthesis):</b>")
        moss_tts_label.setTextFormat(Qt.RichText)
        grid_layout.addWidget(moss_tts_label, 2, 0)

        moss_tts_info = QLabel(
            "<a href='https://huggingface.co/OpenMOSS-Team/MOSS-TTS'>"
            "https://huggingface.co/OpenMOSS-Team/MOSS-TTS</a><br>"
            "Expected: <b>model directory</b> containing config.json and model files<br>"
            "Place in: <b>Models/moss-tts/</b><br>"
            "Also available: Nano, Local-Transformer, or GGUF variants"
        )
        moss_tts_info.setTextFormat(Qt.RichText)
        moss_tts_info.setOpenExternalLinks(True)
        moss_tts_info.setWordWrap(True)
        grid_layout.addWidget(moss_tts_info, 2, 1)

        moss_tts_available = _scan_available(ModelManager.scan_moss_tts_models, "MOSS-TTS")
        self.moss_tts_checkbox = QCheckBox("Available locally")
        self.moss_tts_checkbox.setChecked(moss_tts_available)
        grid_layout.addWidget(self.moss_tts_checkbox, 2, 2)

        layout.addLayout(grid_layout)

        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)

        self.setLayout(layout)
