import sys
import logging
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QProgressBar, QTextEdit,
    QFileDialog, QMessageBox, QDialog, QGridLayout, QGroupBox, QCheckBox
)
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import os
# Add the project root to the path to allow imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.transcriber import Transcriber
from src.core.translator import Translator
from src.core.voice_cloner import VoiceCloner
from src.utils.helpers import (
    scan_model_files, validate_audio_file, validate_reference_audio_duration,
    map_language_code, ensure_directory_exists, get_supported_audio_formats, sanitize_filename
)
from src.config.app_config import get_config


class ModelInfoDialog(QDialog):
    """Modal dialog showing required model downloads with local availability check"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Required Model Downloads")
        self.setModal(True)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Create a grid layout for model information
        grid_layout = QGridLayout()

        # Whisper model section
        whisper_label = QLabel("<b>Whisper:</b>")
        whisper_label.setTextFormat(Qt.RichText)
        grid_layout.addWidget(whisper_label, 0, 0)

        whisper_info = QLabel(
            "<a href='https://huggingface.co/ggerganov/whisper.cpp/tree/main'>https://huggingface.co/ggerganov/whisper.cpp/tree/main</a><br>"
            "Expected extension: <b>.bin</b> (ggml format) or <b>.gguf</b> (GGUF format)<br>"
            "Place in: <b>Models/whisper/</b><br>"
            "Common models: ggml-tiny.bin, ggml-base.bin, ggml-small.bin, ggml-medium.bin, ggml-large.bin"
        )
        whisper_info.setTextFormat(Qt.RichText)
        whisper_info.setOpenExternalLinks(True)
        whisper_info.setWordWrap(True)
        grid_layout.addWidget(whisper_info, 0, 1)

        # Check for Whisper models
        whisper_available = self.check_whisper_models()
        self.whisper_checkbox = QCheckBox("Available locally")
        self.whisper_checkbox.setChecked(whisper_available)
        grid_layout.addWidget(self.whisper_checkbox, 0, 2)

        # NLLB model section
        nllb_label = QLabel("<b>NLLB:</b>")
        nllb_label.setTextFormat(Qt.RichText)
        grid_layout.addWidget(nllb_label, 1, 0)

        nllb_info = QLabel(
            "<a href='https://huggingface.co/facebook/nllb-200-distilled-600M'>https://huggingface.co/facebook/nllb-200-distilled-600M</a><br>"
            "Expected: <b>model directories</b> containing config.json, pytorch_model.bin, tokenizer.json, generation_config.json<br>"
            "Place in: <b>Models/nllb/</b>"
        )
        nllb_info.setTextFormat(Qt.RichText)
        nllb_info.setOpenExternalLinks(True)
        nllb_info.setWordWrap(True)
        grid_layout.addWidget(nllb_info, 1, 1)

        # Check for NLLB models
        nllb_available = self.check_nllb_models()
        self.nllb_checkbox = QCheckBox("Available locally")
        self.nllb_checkbox.setChecked(nllb_available)
        grid_layout.addWidget(self.nllb_checkbox, 1, 2)

        # XTTS-v2 model section
        xtts_label = QLabel("<b>XTTS-v2:</b>")
        xtts_label.setTextFormat(Qt.RichText)
        grid_layout.addWidget(xtts_label, 2, 0)

        xtts_info = QLabel(
            "<a href='https://huggingface.co/coqui/XTTS-v2'>https://huggingface.co/coqui/XTTS-v2</a><br>"
            "Expected: <b>model directories</b> containing config.json, model.pth, vocab.json, speakers.pth, language_ids.json<br>"
            "Place in: <b>Models/xtts/</b>"
        )
        xtts_info.setTextFormat(Qt.RichText)
        xtts_info.setOpenExternalLinks(True)
        xtts_info.setWordWrap(True)
        grid_layout.addWidget(xtts_info, 2, 1)

        # Check for XTTS models
        xtts_available = self.check_xtts_models()
        self.xtts_checkbox = QCheckBox("Available locally")
        self.xtts_checkbox.setChecked(xtts_available)
        grid_layout.addWidget(self.xtts_checkbox, 2, 2)

        layout.addLayout(grid_layout)

        # Refresh button to recheck model availability
        refresh_button = QPushButton("Refresh Model Status")
        refresh_button.clicked.connect(self.refresh_model_status)
        layout.addWidget(refresh_button)

        # Instructions
        instructions = QLabel(
            "<p><i>Check the boxes above if models are available locally.</i></p>"
            "<p><i>Click OK when models are ready.</i></p>"
        )
        instructions.setTextFormat(Qt.RichText)
        layout.addWidget(instructions)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        button_layout.addWidget(ok_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def check_whisper_models(self):
        """Check if Whisper models are available locally"""
        whisper_dir = "./Models/whisper"
        if os.path.exists(whisper_dir):
            # Look for common Whisper model extensions in the whisper directory
            whisper_extensions = ['.bin', '.gguf']
            for file in os.listdir(whisper_dir):
                if any(file.lower().endswith(ext) for ext in whisper_extensions):
                    return True
        return False

    def check_nllb_models(self):
        """Check if NLLB models are available locally"""
        nllb_dir = "./Models/nllb"
        if os.path.exists(nllb_dir):
            # First check if required files exist directly in the nllb directory
            nllb_required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json', 'generation_config.json']
            direct_files_count = 0
            for required_file in nllb_required_files:
                if os.path.exists(os.path.join(nllb_dir, required_file)):
                    direct_files_count += 1
            # If at least 1 of the key required files exist directly in the directory, consider it valid
            # pytorch_model.bin is the most important file for NLLB
            if direct_files_count >= 1 and os.path.exists(os.path.join(nllb_dir, 'pytorch_model.bin')):
                return True
            # Or if at least 2 of the required files exist directly in the directory, consider it valid
            elif direct_files_count >= 2:
                return True
            
            # Also check for directories in the nllb directory that contain model files
            for item in os.listdir(nllb_dir):
                item_path = os.path.join(nllb_dir, item)
                if os.path.isdir(item_path):
                    # Check if directory contains common NLLB model files
                    model_files_count = 0
                    for required_file in nllb_required_files:
                        if os.path.exists(os.path.join(item_path, required_file)):
                            model_files_count += 1
                    # If at least 1 of the key required files exist in the subdirectory, consider it valid
                    # pytorch_model.bin is the most important file for NLLB
                    if model_files_count >= 1 and os.path.exists(os.path.join(item_path, 'pytorch_model.bin')):
                        return True
                    # Or if at least 2 of the required files exist, consider it a valid model
                    elif model_files_count >= 2:
                        return True
        return False

    def check_xtts_models(self):
        """Check if XTTS models are available locally"""
        xtts_dir = "./Models/xtts"
        if os.path.exists(xtts_dir):
            # First check if required files exist directly in the xtts directory
            xtts_required_files = ['config.json', 'model.pth', 'vocab.json', 'speakers.pth', 'language_ids.json']
            direct_files_count = 0
            for required_file in xtts_required_files:
                if os.path.exists(os.path.join(xtts_dir, required_file)):
                    direct_files_count += 1
            # If at least 1 of the key required files exist directly in the directory, consider it valid
            # model.pth is the most important file for XTTS
            if direct_files_count >= 1 and os.path.exists(os.path.join(xtts_dir, 'model.pth')):
                return True
            # Or if at least 2 of the required files exist directly in the directory, consider it valid
            elif direct_files_count >= 2:
                return True
            
            # Also check for directories in the xtts directory that contain model files
            for item in os.listdir(xtts_dir):
                item_path = os.path.join(xtts_dir, item)
                if os.path.isdir(item_path):
                    # Check if directory contains common XTTS model files
                    model_files_count = 0
                    for required_file in xtts_required_files:
                        if os.path.exists(os.path.join(item_path, required_file)):
                            model_files_count += 1
                    # If at least 1 of the key required files exist in the subdirectory, consider it valid
                    # model.pth is the most important file for XTTS
                    if model_files_count >= 1 and os.path.exists(os.path.join(item_path, 'model.pth')):
                        return True
                    # Or if at least 2 of the required files exist, consider it a valid model
                    elif model_files_count >= 2:
                        return True
        return False

    def refresh_model_status(self):
        """Refresh the status of all model availability checks"""
        self.whisper_checkbox.setChecked(self.check_whisper_models())
        self.nllb_checkbox.setChecked(self.check_nllb_models())
        self.xtts_checkbox.setChecked(self.check_xtts_models())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Local Audio Dubbing")
        self.setGeometry(100, 100, 900, 700)

        # Set up logging
        logging.basicConfig(
            level=logging.DEBUG,  # Changed to DEBUG for more detailed logging
            format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            handlers=[
                logging.FileHandler('offline_dubbing.log'),
                logging.StreamHandler()
            ]
        )
        logging.info("Application started")

        # Set Times New Roman font
        font = QFont("Times New Roman", 10)
        self.setFont(font)

        # Set teal color scheme
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(240, 248, 255))  # Light background
        palette.setColor(QPalette.Button, QColor(0, 128, 128))     # Teal buttons
        palette.setColor(QPalette.Highlight, QColor(0, 128, 128)) # Teal highlight
        self.setPalette(palette)

        # Center the window
        self.center_window()

        # Initialize UI components
        self.init_ui()

        # Check if all models are available and conditionally show model info dialog
        if not self.all_models_available():
            self.show_model_info_dialog()
    
    def center_window(self):
        """Center the window on the screen"""
        qr = self.frameGeometry()
        cp = self.screen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def all_models_available(self):
        """Check if all required models are available locally"""
        # Check for Whisper models
        whisper_available = False
        whisper_dir = "./Models/whisper"
        if os.path.exists(whisper_dir):
            # Look for common Whisper model extensions in the whisper directory
            whisper_extensions = ['.bin', '.gguf']
            for file in os.listdir(whisper_dir):
                if any(file.lower().endswith(ext) for ext in whisper_extensions):
                    whisper_available = True
                    break

        # Check for NLLB models
        nllb_available = False
        nllb_dir = "./Models/nllb"
        if os.path.exists(nllb_dir):
            # First check if required files exist directly in the nllb directory
            nllb_required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json', 'generation_config.json']
            direct_files_count = 0
            for required_file in nllb_required_files:
                if os.path.exists(os.path.join(nllb_dir, required_file)):
                    direct_files_count += 1
            # If pytorch_model.bin is present or at least 2 required files exist, consider it valid
            if direct_files_count >= 1 and os.path.exists(os.path.join(nllb_dir, 'pytorch_model.bin')):
                nllb_available = True
            elif direct_files_count >= 2:
                nllb_available = True
            
            # Also check for directories in the nllb directory that contain model files
            if not nllb_available:
                for item in os.listdir(nllb_dir):
                    item_path = os.path.join(nllb_dir, item)
                    if os.path.isdir(item_path):
                        # Check if directory contains common NLLB model files
                        model_files_count = 0
                        for required_file in nllb_required_files:
                            if os.path.exists(os.path.join(item_path, required_file)):
                                model_files_count += 1
                        # If pytorch_model.bin is present or at least 2 required files exist, consider it valid
                        if model_files_count >= 1 and os.path.exists(os.path.join(item_path, 'pytorch_model.bin')):
                            nllb_available = True
                            break
                        elif model_files_count >= 2:
                            nllb_available = True
                            break

        # Check for XTTS models
        xtts_available = False
        xtts_dir = "./Models/xtts"
        if os.path.exists(xtts_dir):
            # First check if required files exist directly in the xtts directory
            xtts_required_files = ['config.json', 'model.pth', 'vocab.json', 'speakers.pth', 'language_ids.json']
            direct_files_count = 0
            for required_file in xtts_required_files:
                if os.path.exists(os.path.join(xtts_dir, required_file)):
                    direct_files_count += 1
            # If model.pth is present or at least 2 required files exist, consider it valid
            if direct_files_count >= 1 and os.path.exists(os.path.join(xtts_dir, 'model.pth')):
                xtts_available = True
            elif direct_files_count >= 2:
                xtts_available = True
            
            # Also check for directories in the xtts directory that contain model files
            if not xtts_available:
                for item in os.listdir(xtts_dir):
                    item_path = os.path.join(xtts_dir, item)
                    if os.path.isdir(item_path):
                        # Check if directory contains common XTTS model files
                        model_files_count = 0
                        for required_file in xtts_required_files:
                            if os.path.exists(os.path.join(item_path, required_file)):
                                model_files_count += 1
                        # If model.pth is present or at least 2 required files exist, consider it valid
                        if model_files_count >= 1 and os.path.exists(os.path.join(item_path, 'model.pth')):
                            xtts_available = True
                            break
                        elif model_files_count >= 2:
                            xtts_available = True
                            break

        return whisper_available and nllb_available and xtts_available

    def show_model_info_dialog(self):
        """Show the model info dialog"""
        dialog = ModelInfoDialog(self)
        dialog.exec_()
    
    def init_ui(self):
        """Initialize the main UI components"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignTop)
        
        # Model Selection Section
        model_group = QGroupBox("Model Selection")
        model_layout = QGridLayout()
        
        # Transcription model selection (formerly Whisper)
        transcription_label = QLabel("Transcription Model:")
        transcription_label.setToolTip("Select the model for speech recognition")
        self.whisper_combo = QComboBox()
        self.whisper_combo.setToolTip("Choose a model for transcription")
        self.refresh_whisper_models()

        refresh_models_btn = QPushButton("Refresh Models")
        refresh_models_btn.setToolTip("Scan for newly added model files")
        refresh_models_btn.clicked.connect(self.refresh_all_models)

        model_layout.addWidget(transcription_label, 0, 0)
        model_layout.addWidget(self.whisper_combo, 0, 1)

        # Translation model selection (formerly NLLB)
        translation_label = QLabel("Translation Model:")
        translation_label.setToolTip("Select the model for translation")
        self.nllb_combo = QComboBox()
        self.nllb_combo.setToolTip("Choose a model for language translation")
        self.refresh_nllb_models()
        model_layout.addWidget(translation_label, 1, 0)
        model_layout.addWidget(self.nllb_combo, 1, 1)

        # Narration model selection (formerly XTTS)
        narration_label = QLabel("Narration Model:")
        narration_label.setToolTip("Select the model for voice synthesis")
        self.xtts_combo = QComboBox()
        self.xtts_combo.setToolTip("Choose a model for voice synthesis")
        self.refresh_xtts_models()
        model_layout.addWidget(narration_label, 2, 0)
        model_layout.addWidget(self.xtts_combo, 2, 1)
        
        model_layout.addWidget(refresh_models_btn, 3, 0, 1, 2, Qt.AlignRight)
        
        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)
        
        # Input Section
        input_group = QGroupBox("Input Files")
        input_layout = QGridLayout()

        # Audio file selection
        audio_label = QLabel("Audio File:")
        audio_label.setToolTip("Select the audio file to be processed")
        audio_btn = QPushButton("Browse...")
        audio_btn.setToolTip("Select an audio file to process")
        audio_btn.clicked.connect(self.select_audio_file)
        self.audio_input = QLabel("(No file selected)")
        self.audio_input.setToolTip("Current selected audio file for processing")

        input_layout.addWidget(audio_label, 0, 0)
        input_layout.addWidget(audio_btn, 0, 1)
        input_layout.addWidget(self.audio_input, 0, 2)

        # Reference audio selection
        ref_label = QLabel("Reference Audio:")
        ref_label.setToolTip("Select a reference audio file for voice cloning (6-10 seconds recommended)")
        ref_btn = QPushButton("Browse...")
        ref_btn.setToolTip("Select a reference audio file for voice cloning")
        ref_btn.clicked.connect(self.select_ref_audio_file)
        self.ref_input = QLabel("(No file selected)")
        self.ref_input.setToolTip("Current selected reference audio for voice cloning")

        input_layout.addWidget(ref_label, 1, 0)
        input_layout.addWidget(ref_btn, 1, 1)
        input_layout.addWidget(self.ref_input, 1, 2)

        # Source language selection
        src_lang_label = QLabel("Source Language:")
        src_lang_label.setToolTip("Select the language of the input audio or choose auto-detect")
        self.src_lang_combo = QComboBox()
        self.populate_language_combo(self.src_lang_combo)
        self.src_lang_combo.addItem("Auto-detect", "auto")
        self.src_lang_combo.setToolTip("Choose the source language or detect automatically")
        
        input_layout.addWidget(src_lang_label, 2, 0)
        input_layout.addWidget(self.src_lang_combo, 2, 1, 1, 2)

        # Target language selection
        tgt_lang_label = QLabel("Target Language(s):")
        tgt_lang_label.setToolTip("Select the language(s) to translate the audio to")
        self.tgt_lang_combo = QComboBox()
        self.tgt_lang_combo.setToolTip("Choose the target language for translation")
        self.populate_language_combo(self.tgt_lang_combo)
        
        input_layout.addWidget(tgt_lang_label, 3, 0)
        input_layout.addWidget(self.tgt_lang_combo, 3, 1, 1, 2)

        input_group.setLayout(input_layout)
        main_layout.addWidget(input_group)

        # Mode Selection - now triggers processing directly
        mode_group = QGroupBox("Processing Mode")
        mode_layout = QHBoxLayout()

        self.transcription_only_btn = QPushButton("Transcription Only")
        self.transcription_only_btn.setCheckable(False)  # No longer checkable, just a trigger
        self.transcription_only_btn.setToolTip("Generate only the text transcription of the audio")
        self.transcription_only_btn.clicked.connect(lambda: self.start_processing(transcription_only=True))

        self.dubbed_translation_btn = QPushButton("Dubbed Translation")
        self.dubbed_translation_btn.setCheckable(False)  # No longer checkable, just a trigger
        self.dubbed_translation_btn.setToolTip("Translate the audio and generate dubbed speech with cloned voice")
        
        # Connect the dubbed translation button to start processing with transcription_only=False
        self.dubbed_translation_btn.clicked.connect(lambda: self.start_processing(transcription_only=False))

        mode_layout.addWidget(self.transcription_only_btn)
        mode_layout.addWidget(self.dubbed_translation_btn)
        mode_group.setLayout(mode_layout)
        main_layout.addWidget(mode_group)

        # Progress Section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setToolTip("Shows the progress of the audio processing")
        self.status_label = QLabel("Ready to process")
        self.status_label.setToolTip("Current status of the processing operation")
        self.log_area = QTextEdit()
        self.log_area.setMaximumHeight(150)
        self.log_area.setReadOnly(True)
        self.log_area.setToolTip("Log of processing events and messages")

        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.status_label)
        progress_layout.addWidget(self.log_area)

        progress_group.setLayout(progress_layout)
        main_layout.addWidget(progress_group)
        
        central_widget.setLayout(main_layout)
        
        # Initialize variables
        self.audio_file_path = ""
        self.ref_audio_path = ""
        self.processing_thread = None
    
    def populate_language_combo(self, combo):
        """Populate language combo box with common languages"""
        languages = [
            ("English", "eng_Latn"),
            ("Spanish", "spa_Latn"),
            ("French", "fra_Latn"),
            ("German", "deu_Latn"),
            ("Italian", "ita_Latn"),
            ("Portuguese", "por_Latn"),
            ("Russian", "rus_Cyrl"),
            ("Chinese", "zho_Hans"),
            ("Japanese", "jpn_Jpan"),
            ("Korean", "kor_Hang"),
            ("Arabic", "ara_Arab"),
            ("Hindi", "hin_Deva")
        ]
        
        for name, code in languages:
            combo.addItem(name, code)
    
    def refresh_all_models(self):
        """Refresh all model dropdowns"""
        self.refresh_whisper_models()
        self.refresh_nllb_models()
        self.refresh_xtts_models()
    
    def refresh_whisper_models(self):
        """Refresh Whisper model dropdown"""
        try:
            self.whisper_combo.clear()
            
            # Look for both .bin and .gguf files in the whisper directory
            whisper_dir = "./Models/whisper"
            if os.path.exists(whisper_dir):
                # Look for both .bin and .gguf files
                all_models = scan_model_files(whisper_dir, [".bin", ".gguf"])
                
                for model in all_models:
                    model_name = os.path.basename(model)
                    self.whisper_combo.addItem(model_name, model)

            if self.whisper_combo.count() == 0:
                self.whisper_combo.addItem("No models found", "")
                logging.warning("No Whisper models found in Models/whisper directory")
            else:
                logging.info(f"Loaded {self.whisper_combo.count()} Whisper models")
        except Exception as e:
            logging.error(f"Error refreshing Whisper models: {str(e)}")
            self.whisper_combo.clear()
            self.whisper_combo.addItem("Error loading models", "")
            QMessageBox.critical(self, "Error", f"Failed to load Whisper models: {str(e)}")

    def refresh_nllb_models(self):
        """Refresh NLLB model dropdown"""
        try:
            self.nllb_combo.clear()
            nllb_dir = "./Models/nllb"
            
            # Check if the main NLLB directory itself contains model files
            nllb_main_valid = False
            if os.path.exists(nllb_dir):
                nllb_required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json', 'generation_config.json']
                direct_files_count = 0
                for required_file in nllb_required_files:
                    if os.path.exists(os.path.join(nllb_dir, required_file)):
                        direct_files_count += 1
                
                # If pytorch_model.bin is present or at least 2 required files exist, consider main directory as valid
                if direct_files_count >= 1 and os.path.exists(os.path.join(nllb_dir, 'pytorch_model.bin')):
                    self.nllb_combo.addItem("NLLB Model", nllb_dir)
                    nllb_main_valid = True
                elif direct_files_count >= 2:
                    self.nllb_combo.addItem("NLLB Model", nllb_dir)
                    nllb_main_valid = True
                
                # Also look for subdirectories that contain model files
                for item in os.listdir(nllb_dir):
                    item_path = os.path.join(nllb_dir, item)
                    if os.path.isdir(item_path):
                        # Check if directory contains common NLLB model files
                        model_files_count = 0
                        for required_file in nllb_required_files:
                            if os.path.exists(os.path.join(item_path, required_file)):
                                model_files_count += 1
                        
                        # If pytorch_model.bin is present or at least 2 required files exist, add to combo
                        if model_files_count >= 1 and os.path.exists(os.path.join(item_path, 'pytorch_model.bin')):
                            self.nllb_combo.addItem(item, item_path)
                        elif model_files_count >= 2:
                            self.nllb_combo.addItem(item, item_path)
            
            if self.nllb_combo.count() == 0:
                self.nllb_combo.addItem("No models found", "")
                logging.warning("No NLLB models found in Models/nllb directory")
            else:
                logging.info(f"Loaded {self.nllb_combo.count()} NLLB model(s) from {nllb_dir}")
        except Exception as e:
            logging.error(f"Error refreshing NLLB models: {str(e)}")
            self.nllb_combo.clear()
            self.nllb_combo.addItem("Error loading models", "")
            QMessageBox.critical(self, "Error", f"Failed to load NLLB models: {str(e)}")

    def refresh_xtts_models(self):
        """Refresh XTTS model dropdown"""
        try:
            self.xtts_combo.clear()
            xtts_dir = "./Models/xtts"
            
            # Check if the main XTTS directory itself contains model files
            xtts_main_valid = False
            if os.path.exists(xtts_dir):
                xtts_required_files = ['config.json', 'model.pth', 'vocab.json', 'speakers.pth', 'language_ids.json']
                direct_files_count = 0
                for required_file in xtts_required_files:
                    if os.path.exists(os.path.join(xtts_dir, required_file)):
                        direct_files_count += 1
                
                # If model.pth is present or at least 2 required files exist, consider main directory as valid
                if direct_files_count >= 1 and os.path.exists(os.path.join(xtts_dir, 'model.pth')):
                    self.xtts_combo.addItem("XTTS Model", xtts_dir)
                    xtts_main_valid = True
                elif direct_files_count >= 2:
                    self.xtts_combo.addItem("XTTS Model", xtts_dir)
                    xtts_main_valid = True
                
                # Also look for subdirectories that contain model files
                for item in os.listdir(xtts_dir):
                    item_path = os.path.join(xtts_dir, item)
                    if os.path.isdir(item_path):
                        # Check if directory contains common XTTS model files
                        model_files_count = 0
                        for required_file in xtts_required_files:
                            if os.path.exists(os.path.join(item_path, required_file)):
                                model_files_count += 1
                        
                        # If model.pth is present or at least 2 required files exist, add to combo
                        if model_files_count >= 1 and os.path.exists(os.path.join(item_path, 'model.pth')):
                            self.xtts_combo.addItem(item, item_path)
                        elif model_files_count >= 2:
                            self.xtts_combo.addItem(item, item_path)
            
            if self.xtts_combo.count() == 0:
                self.xtts_combo.addItem("No models found", "")
                logging.warning("No XTTS models found in Models/xtts directory")
            else:
                logging.info(f"Loaded {self.xtts_combo.count()} XTTS model(s) from {xtts_dir}")
        except Exception as e:
            logging.error(f"Error refreshing XTTS models: {str(e)}")
            self.xtts_combo.clear()
            self.xtts_combo.addItem("Error loading models", "")
            QMessageBox.critical(self, "Error", f"Failed to load XTTS models: {str(e)}")
    
    def select_audio_file(self):
        """Open file dialog to select audio file"""
        file_filter = "Audio Files (" + " ".join([f"*{ext}" for ext in get_supported_audio_formats()]) + ")"
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "", file_filter
        )
        
        if file_path:
            self.audio_file_path = file_path
            self.audio_input.setText(os.path.basename(file_path))
            self.log_message(f"Selected audio file: {file_path}")
    
    def select_ref_audio_file(self):
        """Open file dialog to select reference audio file"""
        file_filter = "Audio Files (" + " ".join([f"*{ext}" for ext in get_supported_audio_formats()]) + ")"
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Audio File", "", file_filter
        )
        
        if file_path:
            is_valid, duration, msg = validate_reference_audio_duration(file_path)
            if is_valid:
                self.ref_audio_path = file_path
                self.ref_input.setText(os.path.basename(file_path))
                self.log_message(f"Selected reference audio: {file_path} ({duration:.2f}s)")
            else:
                QMessageBox.warning(self, "Invalid Reference Audio", msg)
    
    def start_processing(self, transcription_only=False):
        """Start the audio processing"""
        # Validate inputs
        if not self.validate_inputs(transcription_only):
            return

        # Start processing in a separate thread
        self.processing_thread = ProcessingThread(
            self.audio_file_path,
            self.ref_audio_path if not transcription_only else None,
            self.whisper_combo.currentData(),
            self.nllb_combo.currentData(),
            self.xtts_combo.currentData(),
            self.src_lang_combo.currentData(),
            self.tgt_lang_combo.currentData(),
            transcription_only
        )

        # Connect signals
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.status_updated.connect(self.update_status)
        self.processing_thread.log_updated.connect(self.log_message)
        self.processing_thread.processing_finished.connect(self.on_processing_finished)

        # Start the thread
        self.processing_thread.start()
    
    def validate_inputs(self, transcription_only=False):
        """Validate all input fields"""
        errors = []

        try:
            # Check if models are selected
            if self.whisper_combo.currentData() == "":
                errors.append("Please select a Transcription model")
            elif not os.path.exists(self.whisper_combo.currentData()):
                errors.append("Selected Transcription model file does not exist")

            if not transcription_only:
                if self.nllb_combo.currentData() == "":
                    errors.append("Please select a Translation model")
                elif not os.path.exists(self.nllb_combo.currentData()):
                    errors.append("Selected Translation model directory does not exist")
                else:
                    # Validate that the selected NLLB directory has required model files
                    nllb_required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json', 'generation_config.json']
                    required_found = 0
                    for required_file in nllb_required_files:
                        if os.path.exists(os.path.join(self.nllb_combo.currentData(), required_file)):
                            required_found += 1

                    # Check if pytorch_model.bin exists (key file) or at least 2 required files exist
                    if required_found < 1 or (required_found < 2 and not os.path.exists(os.path.join(self.nllb_combo.currentData(), 'pytorch_model.bin'))):
                        errors.append("Selected Translation model directory is missing required model files (config.json, pytorch_model.bin, etc.)")

                if self.xtts_combo.currentData() == "":
                    errors.append("Please select a Narration model")
                elif not os.path.exists(self.xtts_combo.currentData()):
                    errors.append("Selected Narration model directory does not exist")
                else:
                    # Validate that the selected XTTS directory has required model files
                    xtts_required_files = ['config.json', 'model.pth', 'vocab.json', 'speakers.pth', 'language_ids.json']
                    required_found = 0
                    for required_file in xtts_required_files:
                        if os.path.exists(os.path.join(self.xtts_combo.currentData(), required_file)):
                            required_found += 1

                    # Check if model.pth exists (key file) or at least 2 required files exist
                    if required_found < 1 or (required_found < 2 and not os.path.exists(os.path.join(self.xtts_combo.currentData(), 'model.pth'))):
                        errors.append("Selected Narration model directory is missing required model files (config.json, model.pth, etc.)")

            # Check if audio file is selected
            if not self.audio_file_path:
                errors.append("Please select an audio file")
            elif not os.path.exists(self.audio_file_path):
                errors.append("Selected audio file does not exist")
            elif not validate_audio_file(self.audio_file_path):
                errors.append("Selected audio file is invalid")

            # Check if reference audio is valid when in dubbed translation mode
            if not transcription_only:
                if not self.ref_audio_path:
                    errors.append("Please select a reference audio file for voice cloning")
                elif not os.path.exists(self.ref_audio_path):
                    errors.append("Selected reference audio file does not exist")
                elif not validate_audio_file(self.ref_audio_path):
                    errors.append("Selected reference audio file is invalid")
                else:
                    is_valid, duration, msg = validate_reference_audio_duration(self.ref_audio_path)
                    if not is_valid:
                        errors.append(msg)

            # Show errors if any
            if errors:
                error_msg = "\n".join(errors)
                logging.error(f"Input validation failed: {error_msg}")
                QMessageBox.critical(self, "Validation Error", f"The following errors occurred:\n\n{error_msg}")
                return False

            logging.info("Input validation passed")
            return True
        except Exception as e:
            logging.error(f"Error during input validation: {str(e)}")
            QMessageBox.critical(self, "Validation Error", f"An error occurred during validation: {str(e)}")
            return False
    
    def update_progress(self, value):
        """Update the progress bar"""
        self.progress_bar.setValue(value)
    
    def update_status(self, status):
        """Update the status label"""
        self.status_label.setText(status)
    
    def log_message(self, message):
        """Add a message to the log area"""
        self.log_area.append(message)

    def on_processing_finished(self, success, message):
        """Handle processing completion"""
        # Re-enable the mode buttons after processing is complete
        self.transcription_only_btn.setEnabled(True)
        self.dubbed_translation_btn.setEnabled(True)

        if success:
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.critical(self, "Error", message)


class ProcessingThread(QThread):
    """Thread for processing audio to keep UI responsive"""
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    log_updated = pyqtSignal(str)
    processing_finished = pyqtSignal(bool, str)

    def __init__(self, audio_file, ref_audio, whisper_model, nllb_model, xtts_model,
                 src_lang, tgt_lang, transcription_only):
        super().__init__()
        self.audio_file = audio_file
        self.ref_audio = ref_audio
        self.whisper_model = whisper_model
        self.nllb_model = nllb_model
        self.xtts_model = xtts_model
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.transcription_only = transcription_only
        self._stop_flag = False

    def stop(self):
        """Set the stop flag to terminate processing"""
        self._stop_flag = True

    def run(self):
        """Main processing logic"""
        try:
            logging.info("Starting audio processing thread")
            # Ensure output directory exists
            ensure_directory_exists("./Outputs")

            self.status_updated.emit("Loading Whisper model...")
            self.log_updated.emit("Loading Whisper model...")

            if self._stop_flag:
                logging.info("Processing stopped by user")
                self.processing_finished.emit(False, "Processing stopped by user")
                return

            # Initialize transcriber
            self.log_updated.emit(f"Initializing Whisper model: {self.whisper_model}")
            logging.debug(f"Whisper model path: {self.whisper_model}")
            transcriber = Transcriber(self.whisper_model)

            if self._stop_flag:
                logging.info("Processing stopped by user")
                self.processing_finished.emit(False, "Processing stopped by user")
                return

            self.status_updated.emit("Transcribing audio...")
            self.log_updated.emit("Starting transcription...")
            self.progress_updated.emit(10)

            # Transcribe audio
            self.log_updated.emit(f"Transcribing audio file: {self.audio_file}")
            logging.debug(f"Audio file path: {self.audio_file}, Source language: {self.src_lang}")
            transcription_result = transcriber.transcribe(self.audio_file, self.src_lang if self.src_lang != "auto" else None)
            transcribed_text = transcription_result["text"]
            detected_language = transcription_result["language"]

            self.log_updated.emit(f"Transcription completed. Detected language: {detected_language}")
            logging.debug(f"Transcribed text length: {len(transcribed_text)} characters")
            self.progress_updated.emit(30)

            # If transcription only mode, save and exit
            if self.transcription_only:
                logging.info("Running in transcription-only mode")
                # Sanitize the output filename to prevent issues
                sanitized_name = sanitize_filename(os.path.splitext(os.path.basename(self.audio_file))[0])
                output_path = f"./Outputs/{sanitized_name}_transcript.txt"

                self.log_updated.emit(f"Saving transcription to: {output_path}")
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(transcribed_text)

                self.log_updated.emit(f"Transcription saved to: {output_path}")
                self.status_updated.emit("Transcription completed successfully!")
                self.progress_updated.emit(100)
                self.processing_finished.emit(True, f"Transcription saved to: {output_path}")
                logging.info(f"Transcription saved to: {output_path}")
                return

            if self._stop_flag:
                logging.info("Processing stopped by user")
                self.processing_finished.emit(False, "Processing stopped by user")
                return

            self.status_updated.emit("Loading NLLB translator...")
            self.log_updated.emit("Loading NLLB model...")
            self.progress_updated.emit(40)

            # Initialize translator
            self.log_updated.emit(f"Initializing NLLB model: {self.nllb_model}")
            logging.debug(f"NLLB model path: {self.nllb_model}")
            translator = Translator(self.nllb_model)

            if self._stop_flag:
                logging.info("Processing stopped by user")
                self.processing_finished.emit(False, "Processing stopped by user")
                return

            self.status_updated.emit("Translating text...")
            self.log_updated.emit("Starting translation...")
            self.progress_updated.emit(50)

            # Translate text
            self.log_updated.emit(f"Translating from {detected_language} to {self.tgt_lang}")
            logging.debug(f"Source language: {detected_language}, Target language: {self.tgt_lang}")
            translated_text = translator.translate(transcribed_text, detected_language, self.tgt_lang)

            self.log_updated.emit("Translation completed")
            logging.debug(f"Translated text length: {len(translated_text)} characters")
            self.progress_updated.emit(70)

            if self._stop_flag:
                logging.info("Processing stopped by user")
                self.processing_finished.emit(False, "Processing stopped by user")
                return

            self.status_updated.emit("Loading XTTS voice cloner...")
            self.log_updated.emit("Loading XTTS model...")
            self.progress_updated.emit(80)

            # Initialize voice cloner
            self.log_updated.emit(f"Initializing XTTS model: {self.xtts_model}")
            logging.debug(f"XTTS model path: {self.xtts_model}")
            cloner = VoiceCloner(self.xtts_model)

            if self._stop_flag:
                logging.info("Processing stopped by user")
                self.processing_finished.emit(False, "Processing stopped by user")
                return

            self.status_updated.emit("Generating dubbed audio...")
            self.log_updated.emit("Generating audio with cloned voice...")
            self.progress_updated.emit(90)

            # Generate audio with cloned voice
            sanitized_name = sanitize_filename(os.path.splitext(os.path.basename(self.audio_file))[0])
            output_path = f"./Outputs/{sanitized_name}_{self.tgt_lang}.wav"

            self.log_updated.emit(f"Generating dubbed audio: {output_path}")
            logging.debug(f"Reference audio: {self.ref_audio}, Output path: {output_path}, Language: {self.tgt_lang.split('_')[0]}")
            cloner.clone_voice(translated_text, self.ref_audio, output_path, self.tgt_lang.split('_')[0])

            self.log_updated.emit(f"Dubbed audio saved to: {output_path}")
            self.status_updated.emit("Processing completed successfully!")
            self.progress_updated.emit(100)

            self.processing_finished.emit(True, f"Processing completed successfully!\nOutput saved to: {output_path}")
            logging.info(f"Processing completed successfully. Output saved to: {output_path}")

        except FileNotFoundError as e:
            error_msg = f"File not found: {str(e)}"
            self.log_updated.emit(error_msg)
            self.status_updated.emit("Processing failed - file not found!")
            logging.error(f"FileNotFoundError in ProcessingThread: {str(e)}", exc_info=True)
            self.processing_finished.emit(False, error_msg)
        except ValueError as e:
            error_msg = f"Value error: {str(e)}"
            self.log_updated.emit(error_msg)
            self.status_updated.emit("Processing failed - invalid value!")
            logging.error(f"ValueError in ProcessingThread: {str(e)}", exc_info=True)
            self.processing_finished.emit(False, error_msg)
        except RuntimeError as e:
            error_msg = f"Runtime error: {str(e)}"
            self.log_updated.emit(error_msg)
            self.status_updated.emit("Processing failed - runtime error!")
            logging.error(f"RuntimeError in ProcessingThread: {str(e)}", exc_info=True)
            self.processing_finished.emit(False, error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during processing: {str(e)}"
            self.log_updated.emit(error_msg)
            self.status_updated.emit("Processing failed!")
            logging.error(f"Unexpected error in ProcessingThread: {str(e)}", exc_info=True)
            self.processing_finished.emit(False, error_msg)


def main():
    app = QApplication(sys.argv)

    # Set application font
    font = QFont("Times New Roman", 10)
    app.setFont(font)

    try:
        window = MainWindow()
        window.show()
        logging.info("Application started successfully")
        sys.exit(app.exec_())
    except Exception as e:
        logging.error(f"Application error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()