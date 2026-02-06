import sys
import logging
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QProgressBar, QTextEdit,
    QFileDialog, QMessageBox, QDialog, QGridLayout, QGroupBox
)
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import os
from modules.transcriber import Transcriber
from modules.translator import Translator
from modules.voice_cloner import VoiceCloner
from modules.utils import (
    scan_model_files, validate_audio_file, validate_reference_audio_duration,
    map_language_code, ensure_directory_exists, get_supported_audio_formats, sanitize_filename
)


class ModelInfoDialog(QDialog):
    """Modal dialog showing required model downloads"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Required Model Downloads")
        self.setModal(True)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        info_text = """
        <h3>Required Model Downloads:</h3>
        <p><b>Whisper:</b> <a href='https://huggingface.co/ggerganov/whisper.cpp/tree/main'>https://huggingface.co/ggerganov/whisper.cpp/tree/main</a><br>
        Place .bin files (ggml format) in: Models/whisper/<br>
        Common models: ggml-tiny.bin, ggml-base.bin, ggml-small.bin, ggml-medium.bin, ggml-large.bin</p>

        <p><b>NLLB:</b> <a href='https://huggingface.co/facebook/nllb-200-distilled-600M'>https://huggingface.co/facebook/nllb-200-distilled-600M</a><br>
        Place model folder in: Models/nllb/</p>

        <p><b>XTTS-v2:</b> <a href='https://huggingface.co/coqui/XTTS-v2'>https://huggingface.co/coqui/XTTS-v2</a><br>
        Place model files in: Models/xtts/</p>

        <p><i>Click OK when models are ready.</i></p>
        """
        
        label = QLabel(info_text)
        label.setTextFormat(Qt.RichText)
        label.setOpenExternalLinks(True)
        label.setWordWrap(True)
        
        layout.addWidget(label)
        
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        button_layout.addWidget(ok_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Local Audio Dubbing")
        self.setGeometry(100, 100, 900, 700)

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
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

        # Show model info dialog on startup
        self.show_model_info_dialog()
    
    def center_window(self):
        """Center the window on the screen"""
        qr = self.frameGeometry()
        cp = self.screen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
    
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
        
        # Whisper model selection
        whisper_label = QLabel("Whisper Model:")
        whisper_label.setToolTip("Select the Whisper model for speech recognition")
        self.whisper_combo = QComboBox()
        self.whisper_combo.setToolTip("Choose a Whisper model for transcription")
        self.refresh_whisper_models()
        
        refresh_models_btn = QPushButton("Refresh Models")
        refresh_models_btn.setToolTip("Scan for newly added model files")
        refresh_models_btn.clicked.connect(self.refresh_all_models)
        
        model_layout.addWidget(whisper_label, 0, 0)
        model_layout.addWidget(self.whisper_combo, 0, 1)
        
        # NLLB model selection
        nllb_label = QLabel("NLLB Model:")
        nllb_label.setToolTip("Select the NLLB model for translation")
        self.nllb_combo = QComboBox()
        self.nllb_combo.setToolTip("Choose an NLLB model for language translation")
        self.refresh_nllb_models()
        model_layout.addWidget(nllb_label, 1, 0)
        model_layout.addWidget(self.nllb_combo, 1, 1)
        
        # XTTS model selection
        xtts_label = QLabel("XTTS Model:")
        xtts_label.setToolTip("Select the XTTS model for voice cloning")
        self.xtts_combo = QComboBox()
        self.xtts_combo.setToolTip("Choose an XTTS model for voice synthesis")
        self.refresh_xtts_models()
        model_layout.addWidget(xtts_label, 2, 0)
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
        self.audio_input = QLabel("(No file selected)")
        self.audio_input.setToolTip("Current selected audio file for processing")
        audio_btn = QPushButton("Browse...")
        audio_btn.setToolTip("Select an audio file to process")
        audio_btn.clicked.connect(self.select_audio_file)
        
        input_layout.addWidget(audio_label, 0, 0)
        input_layout.addWidget(self.audio_input, 0, 1)
        input_layout.addWidget(audio_btn, 0, 2)
        
        # Reference audio selection
        ref_label = QLabel("Reference Audio:")
        ref_label.setToolTip("Select a reference audio file for voice cloning (6-10 seconds recommended)")
        self.ref_input = QLabel("(No file selected)")
        self.ref_input.setToolTip("Current selected reference audio for voice cloning")
        ref_btn = QPushButton("Browse...")
        ref_btn.setToolTip("Select a reference audio file for voice cloning")
        ref_btn.clicked.connect(self.select_ref_audio_file)
        
        input_layout.addWidget(ref_label, 1, 0)
        input_layout.addWidget(self.ref_input, 1, 1)
        input_layout.addWidget(ref_btn, 1, 2)
        
        # Source language
        src_lang_label = QLabel("Source Language:")
        src_lang_label.setToolTip("Select the language of the input audio or choose auto-detect")
        self.src_lang_combo = QComboBox()
        self.populate_language_combo(self.src_lang_combo)
        self.src_lang_combo.addItem("Auto-detect", "auto")
        self.src_lang_combo.setToolTip("Choose the source language or detect automatically")
        input_layout.addWidget(src_lang_label, 2, 0)
        input_layout.addWidget(self.src_lang_combo, 2, 1, 1, 2)
        
        # Target language
        tgt_lang_label = QLabel("Target Language(s):")
        tgt_lang_label.setToolTip("Select the language(s) to translate the audio to")
        self.tgt_lang_combo = QComboBox()
        self.tgt_lang_combo.setToolTip("Choose the target language for translation")
        self.populate_language_combo(self.tgt_lang_combo)
        input_layout.addWidget(tgt_lang_label, 3, 0)
        input_layout.addWidget(self.tgt_lang_combo, 3, 1, 1, 2)
        
        input_group.setLayout(input_layout)
        main_layout.addWidget(input_group)
        
        # Mode Selection
        mode_group = QGroupBox("Processing Mode")
        mode_layout = QHBoxLayout()
        
        self.transcription_only_btn = QPushButton("Transcription Only")
        self.transcription_only_btn.setCheckable(True)
        self.transcription_only_btn.setToolTip("Generate only the text transcription of the audio")
        self.dubbed_translation_btn = QPushButton("Dubbed Translation")
        self.dubbed_translation_btn.setCheckable(True)
        self.dubbed_translation_btn.setToolTip("Translate the audio and generate dubbed speech with cloned voice")
        self.dubbed_translation_btn.setChecked(True)  # Default selection
        
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
        
        # Action Buttons
        action_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Processing")
        self.start_btn.setStyleSheet("background-color: #008080; color: white;")
        self.start_btn.setToolTip("Begin processing the selected audio file")
        self.start_btn.clicked.connect(self.start_processing)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setToolTip("Stop the current processing operation")
        self.stop_btn.clicked.connect(self.stop_processing)
        
        self.clear_log_btn = QPushButton("Clear Log")
        self.clear_log_btn.setToolTip("Clear the log messages")
        self.clear_log_btn.clicked.connect(self.clear_log)
        
        action_layout.addWidget(self.start_btn)
        action_layout.addWidget(self.stop_btn)
        action_layout.addWidget(self.clear_log_btn)
        action_layout.addStretch()
        
        main_layout.addLayout(action_layout)
        
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
            models = scan_model_files("./Models/whisper", ".bin")  # Changed from .pt to .bin for ggml models
            for model in models:
                self.whisper_combo.addItem(os.path.basename(model), model)

            if self.whisper_combo.count() == 0:
                self.whisper_combo.addItem("No models found", "")
                logging.warning("No Whisper models found in Models/whisper directory")
            else:
                logging.info(f"Loaded {len(models)} Whisper models")
        except Exception as e:
            logging.error(f"Error refreshing Whisper models: {str(e)}")
            self.whisper_combo.clear()
            self.whisper_combo.addItem("Error loading models", "")
            QMessageBox.critical(self, "Error", f"Failed to load Whisper models: {str(e)}")

    def refresh_nllb_models(self):
        """Refresh NLLB model dropdown"""
        try:
            self.nllb_combo.clear()
            # Look for directories in Models/nllb/
            nllb_dir = "./Models/nllb"
            if os.path.exists(nllb_dir):
                for item in os.listdir(nllb_dir):
                    item_path = os.path.join(nllb_dir, item)
                    if os.path.isdir(item_path):
                        self.nllb_combo.addItem(item, item_path)
            else:
                logging.warning("NLLB models directory does not exist")

            if self.nllb_combo.count() == 0:
                self.nllb_combo.addItem("No models found", "")
                logging.warning("No NLLB models found in Models/nllb directory")
            else:
                logging.info(f"Loaded NLLB models from {nllb_dir}")
        except Exception as e:
            logging.error(f"Error refreshing NLLB models: {str(e)}")
            self.nllb_combo.clear()
            self.nllb_combo.addItem("Error loading models", "")
            QMessageBox.critical(self, "Error", f"Failed to load NLLB models: {str(e)}")

    def refresh_xtts_models(self):
        """Refresh XTTS model dropdown"""
        try:
            self.xtts_combo.clear()
            # Look for directories in Models/xtts/
            xtts_dir = "./Models/xtts"
            if os.path.exists(xtts_dir):
                for item in os.listdir(xtts_dir):
                    item_path = os.path.join(xtts_dir, item)
                    if os.path.isdir(item_path):
                        self.xtts_combo.addItem(item, item_path)
            else:
                logging.warning("XTTS models directory does not exist")

            if self.xtts_combo.count() == 0:
                self.xtts_combo.addItem("No models found", "")
                logging.warning("No XTTS models found in Models/xtts directory")
            else:
                logging.info(f"Loaded XTTS models from {xtts_dir}")
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
    
    def start_processing(self):
        """Start the audio processing"""
        # Validate inputs
        if not self.validate_inputs():
            return
        
        # Disable start button and enable stop button
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # Start processing in a separate thread
        self.processing_thread = ProcessingThread(
            self.audio_file_path,
            self.ref_audio_path if self.dubbed_translation_btn.isChecked() else None,
            self.whisper_combo.currentData(),
            self.nllb_combo.currentData(),
            self.xtts_combo.currentData(),
            self.src_lang_combo.currentData(),
            self.tgt_lang_combo.currentData(),
            self.transcription_only_btn.isChecked()
        )
        
        # Connect signals
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.status_updated.connect(self.update_status)
        self.processing_thread.log_updated.connect(self.log_message)
        self.processing_thread.processing_finished.connect(self.on_processing_finished)
        
        # Start the thread
        self.processing_thread.start()
    
    def stop_processing(self):
        """Stop the audio processing"""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.log_message("Processing stopped by user.")
    
    def validate_inputs(self):
        """Validate all input fields"""
        errors = []

        try:
            # Check if models are selected
            if self.whisper_combo.currentData() == "":
                errors.append("Please select a Whisper model")
            elif not os.path.exists(self.whisper_combo.currentData()):
                errors.append("Selected Whisper model file does not exist")
                
            if self.nllb_combo.currentData() == "":
                errors.append("Please select an NLLB model")
            elif not os.path.exists(self.nllb_combo.currentData()):
                errors.append("Selected NLLB model directory does not exist")
                
            if self.xtts_combo.currentData() == "" and self.dubbed_translation_btn.isChecked():
                errors.append("Please select an XTTS model")
            elif self.xtts_combo.currentData() != "" and self.dubbed_translation_btn.isChecked():
                if not os.path.exists(self.xtts_combo.currentData()):
                    errors.append("Selected XTTS model directory does not exist")

            # Check if audio file is selected
            if not self.audio_file_path:
                errors.append("Please select an audio file")
            elif not os.path.exists(self.audio_file_path):
                errors.append("Selected audio file does not exist")
            elif not validate_audio_file(self.audio_file_path):
                errors.append("Selected audio file is invalid")

            # Check if reference audio is valid when in dubbed translation mode
            if self.dubbed_translation_btn.isChecked():
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
    
    def clear_log(self):
        """Clear the log area"""
        self.log_area.clear()
    
    def on_processing_finished(self, success, message):
        """Handle processing completion"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
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
            # Ensure output directory exists
            ensure_directory_exists("./Outputs")

            self.status_updated.emit("Loading Whisper model...")
            self.log_updated.emit("Loading Whisper model...")

            if self._stop_flag:
                self.processing_finished.emit(False, "Processing stopped by user")
                return

            # Initialize transcriber
            self.log_updated.emit(f"Initializing Whisper model: {self.whisper_model}")
            transcriber = Transcriber(self.whisper_model)

            if self._stop_flag:
                self.processing_finished.emit(False, "Processing stopped by user")
                return

            self.status_updated.emit("Transcribing audio...")
            self.log_updated.emit("Starting transcription...")
            self.progress_updated.emit(10)

            # Transcribe audio
            self.log_updated.emit(f"Transcribing audio file: {self.audio_file}")
            transcription_result = transcriber.transcribe(self.audio_file, self.src_lang if self.src_lang != "auto" else None)
            transcribed_text = transcription_result["text"]
            detected_language = transcription_result["language"]

            self.log_updated.emit(f"Transcription completed. Detected language: {detected_language}")
            self.progress_updated.emit(30)

            # If transcription only mode, save and exit
            if self.transcription_only:
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
                return

            if self._stop_flag:
                self.processing_finished.emit(False, "Processing stopped by user")
                return

            self.status_updated.emit("Loading NLLB translator...")
            self.log_updated.emit("Loading NLLB model...")
            self.progress_updated.emit(40)

            # Initialize translator
            self.log_updated.emit(f"Initializing NLLB model: {self.nllb_model}")
            translator = Translator(self.nllb_model)

            if self._stop_flag:
                self.processing_finished.emit(False, "Processing stopped by user")
                return

            self.status_updated.emit("Translating text...")
            self.log_updated.emit("Starting translation...")
            self.progress_updated.emit(50)

            # Translate text
            self.log_updated.emit(f"Translating from {detected_language} to {self.tgt_lang}")
            translated_text = translator.translate(transcribed_text, detected_language, self.tgt_lang)

            self.log_updated.emit("Translation completed")
            self.progress_updated.emit(70)

            if self._stop_flag:
                self.processing_finished.emit(False, "Processing stopped by user")
                return

            self.status_updated.emit("Loading XTTS voice cloner...")
            self.log_updated.emit("Loading XTTS model...")
            self.progress_updated.emit(80)

            # Initialize voice cloner
            self.log_updated.emit(f"Initializing XTTS model: {self.xtts_model}")
            cloner = VoiceCloner(self.xtts_model)

            if self._stop_flag:
                self.processing_finished.emit(False, "Processing stopped by user")
                return

            self.status_updated.emit("Generating dubbed audio...")
            self.log_updated.emit("Generating audio with cloned voice...")
            self.progress_updated.emit(90)

            # Generate audio with cloned voice
            sanitized_name = sanitize_filename(os.path.splitext(os.path.basename(self.audio_file))[0])
            output_path = f"./Outputs/{sanitized_name}_{self.tgt_lang}.wav"
            
            self.log_updated.emit(f"Generating dubbed audio: {output_path}")
            cloner.clone_voice(translated_text, self.ref_audio, output_path, self.tgt_lang.split('_')[0])

            self.log_updated.emit(f"Dubbed audio saved to: {output_path}")
            self.status_updated.emit("Processing completed successfully!")
            self.progress_updated.emit(100)

            self.processing_finished.emit(True, f"Processing completed successfully!\nOutput saved to: {output_path}")

        except FileNotFoundError as e:
            error_msg = f"File not found: {str(e)}"
            self.log_updated.emit(error_msg)
            self.status_updated.emit("Processing failed - file not found!")
            self.processing_finished.emit(False, error_msg)
        except ValueError as e:
            error_msg = f"Value error: {str(e)}"
            self.log_updated.emit(error_msg)
            self.status_updated.emit("Processing failed - invalid value!")
            self.processing_finished.emit(False, error_msg)
        except RuntimeError as e:
            error_msg = f"Runtime error: {str(e)}"
            self.log_updated.emit(error_msg)
            self.status_updated.emit("Processing failed - runtime error!")
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