import sys
import logging
import os
from datetime import datetime
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QProgressBar, QTextEdit,
    QFileDialog, QMessageBox, QGroupBox, QGridLayout
)
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from src.utils.common.helpers import (
    validate_audio_file, validate_reference_audio_duration,
    get_supported_audio_formats
)
from src.api.interfaces.dialogs import ModelInfoDialog
from src.core.services.model_manager import ModelManager
from src.core.application.audio_orchestrator import AudioDubbingOrchestrator
from src.core.data_models.audio_models import AudioProcessingConfig, ProcessingMode


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
        if not ModelManager.all_models_available():
            self.show_model_info_dialog()
    
    def center_window(self):
        """Center the window on the screen"""
        qr = self.frameGeometry()
        cp = self.screen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def all_models_available(self):
        """Legacy check for backward compatibility, now using ModelManager."""
        return ModelManager.all_models_available()

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

        # Reference audio selection (common for all modes)
        ref_label = QLabel("Reference Audio:")
        ref_label.setToolTip("Select a reference audio file for voice cloning (6-10 seconds recommended)")
        ref_btn = QPushButton("Browse...")
        ref_btn.setToolTip("Select a reference audio file for voice cloning")
        ref_btn.clicked.connect(self.select_ref_audio_file)
        self.ref_input = QLabel("(No file selected)")
        self.ref_input.setToolTip("Current selected reference audio for voice cloning")

        input_layout.addWidget(ref_label, 0, 0)
        input_layout.addWidget(ref_btn, 0, 1)
        input_layout.addWidget(self.ref_input, 0, 2)

        # Input Type Selection
        input_type_label = QLabel("Input Type:")
        input_type_label.setToolTip("Select the type of input you want to process")
        self.input_type_combo = QComboBox()
        self.input_type_combo.addItem("Audio File", "audio")
        self.input_type_combo.addItem("Transcription Text", "transcription")
        self.input_type_combo.addItem("Translation Text", "translation")
        self.input_type_combo.setToolTip("Choose the type of input to process")
        self.input_type_combo.currentIndexChanged.connect(self.on_input_type_changed)

        input_layout.addWidget(input_type_label, 1, 0)
        input_layout.addWidget(self.input_type_combo, 1, 1, 1, 2)

        # Audio file selection (visible only when input type is audio)
        self.audio_label = QLabel("Audio File:")
        self.audio_label.setToolTip("Select the audio file to be processed")
        self.audio_btn = QPushButton("Browse...")
        self.audio_btn.setToolTip("Select an audio file to process")
        self.audio_btn.clicked.connect(self.select_audio_file)
        self.audio_input = QLabel("(No file selected)")
        self.audio_input.setToolTip("Current selected audio file for processing")

        input_layout.addWidget(self.audio_label, 2, 0)
        input_layout.addWidget(self.audio_btn, 2, 1)
        input_layout.addWidget(self.audio_input, 2, 2)

        # Text file selection (visible only when input type is transcription or translation)
        self.text_label = QLabel("Text File:")
        self.text_label.setToolTip("Select the text file to be processed")
        self.text_btn = QPushButton("Browse...")
        self.text_btn.setToolTip("Select a text file to process")
        self.text_btn.clicked.connect(self.select_text_file)
        self.text_input = QLabel("(No file selected)")
        self.text_input.setToolTip("Current selected text file for processing")

        input_layout.addWidget(self.text_label, 3, 0)
        input_layout.addWidget(self.text_btn, 3, 1)
        input_layout.addWidget(self.text_input, 3, 2)

        # Source language selection (visible only when input type is audio)
        self.src_lang_label = QLabel("Source Language:")
        self.src_lang_label.setToolTip("Select the language of the input audio or choose auto-detect")
        self.src_lang_combo = QComboBox()
        self.populate_language_combo(self.src_lang_combo)
        self.src_lang_combo.addItem("Auto-detect", "auto")
        self.src_lang_combo.setToolTip("Choose the source language or detect automatically")

        # Target language selection (visible only when input type is audio or transcription)
        self.tgt_lang_label = QLabel("Target Language:")
        self.tgt_lang_label.setToolTip("Select the target language to translate to")
        self.tgt_lang_combo = QComboBox()
        self.populate_language_combo(self.tgt_lang_combo)
        self.tgt_lang_combo.setToolTip("Choose the target language for translation")

        input_layout.addWidget(self.src_lang_label, 4, 0)
        input_layout.addWidget(self.src_lang_combo, 4, 1)
        input_layout.addWidget(self.tgt_lang_label, 4, 2)
        input_layout.addWidget(self.tgt_lang_combo, 4, 3)

        input_group.setLayout(input_layout)
        main_layout.addWidget(input_group)

        # Output Mode Selection
        mode_group = QGroupBox("Output Mode")
        mode_layout = QHBoxLayout()

        self.transcript_only_btn = QPushButton("Transcription Only")
        self.transcript_only_btn.setCheckable(False)
        self.transcript_only_btn.setToolTip("Generate only the text transcription")
        self.transcript_only_btn.clicked.connect(lambda: self.start_processing(output_mode="transcript"))

        self.dubbed_translation_btn = QPushButton("Dubbed Translation")
        self.dubbed_translation_btn.setCheckable(False)
        self.dubbed_translation_btn.setToolTip("Generate dubbed speech with cloned voice")
        self.dubbed_translation_btn.clicked.connect(lambda: self.start_processing(output_mode="translation"))

        mode_layout.addWidget(self.transcript_only_btn)
        mode_layout.addWidget(self.dubbed_translation_btn)
        mode_group.setLayout(mode_layout)
        main_layout.addWidget(mode_group)

        # Initially hide text-related widgets
        self.text_label.hide()
        self.text_btn.hide()
        self.text_input.hide()
        self.src_lang_label.hide()
        self.src_lang_combo.hide()
        self.tgt_lang_label.hide()
        self.tgt_lang_combo.hide()

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
        
        # Set initial visibility state based on default input type
        self.on_input_type_changed()
        
        # Initialize variables
        self.audio_file_path = ""
        self.ref_audio_path = ""
        self.processing_thread = None
    
    def populate_language_combo(self, combo):
        """Populate language combo box with all NLLB-200 supported languages"""
        # Import the function to get NLLB languages
        from src.utils.common.helpers import get_nllb_languages
        
        languages = get_nllb_languages()
        
        # Sort languages alphabetically by name
        sorted_languages = sorted(languages.items())
        
        for name, code in sorted_languages:
            combo.addItem(name, code)
        
        # Set English as default selection
        index = combo.findData("eng_Latn")
        if index >= 0:
            combo.setCurrentIndex(index)

    def populate_language_checkboxes_horizontal(self):
        """This method is deprecated - target language is now a dropdown combo"""
        # Legacy method kept for backward compatibility but not used
        pass
    
    def refresh_all_models(self):
        """Refresh all model dropdowns"""
        self.refresh_whisper_models()
        self.refresh_nllb_models()
        self.refresh_xtts_models()
    
    def refresh_whisper_models(self):
        """Refresh Whisper model dropdown"""
        try:
            self.whisper_combo.clear()
            models = ModelManager.scan_whisper_models()
            for model in models:
                self.whisper_combo.addItem(model["name"], model["path"])

            if self.whisper_combo.count() == 0:
                self.whisper_combo.addItem("No models found", "")
                logging.warning("No Whisper models found")
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
            models = ModelManager.scan_nllb_models()
            for model in models:
                self.nllb_combo.addItem(model["name"], model["path"])
            
            if self.nllb_combo.count() == 0:
                self.nllb_combo.addItem("No models found", "")
                logging.warning("No NLLB models found")
            else:
                logging.info(f"Loaded {self.nllb_combo.count()} NLLB model(s)")
        except Exception as e:
            logging.error(f"Error refreshing NLLB models: {str(e)}")
            self.nllb_combo.clear()
            self.nllb_combo.addItem("Error loading models", "")
            QMessageBox.critical(self, "Error", f"Failed to load NLLB models: {str(e)}")

    def refresh_xtts_models(self):
        """Refresh XTTS model dropdown"""
        try:
            self.xtts_combo.clear()
            models = ModelManager.scan_xtts_models()
            for model in models:
                self.xtts_combo.addItem(model["name"], model["path"])
            
            if self.xtts_combo.count() == 0:
                self.xtts_combo.addItem("No models found", "")
                logging.warning("No XTTS models found")
            else:
                logging.info(f"Loaded {self.xtts_combo.count()} XTTS model(s)")
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

    def select_text_file(self):
        """Open file dialog to select text file"""
        file_filter = "Text Files (*.txt);;All Files (*)"
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Text File", "", file_filter
        )

        if file_path:
            self.text_file_path = file_path
            self.text_input.setText(os.path.basename(file_path))
            self.log_message(f"Selected text file: {file_path}")

    def on_input_type_changed(self):
        """Handle changes in input type selection"""
        input_type = self.input_type_combo.currentData()
        
        # Show/hide appropriate widgets based on input type
        if input_type == "audio":
            # Show audio file selection
            self.audio_label.show()
            self.audio_btn.show()
            self.audio_input.show()
            
            # Hide text file selection
            self.text_label.hide()
            self.text_btn.hide()
            self.text_input.hide()
            
            # Show source and target language selection
            self.src_lang_label.show()
            self.src_lang_combo.show()
            self.tgt_lang_label.show()
            self.tgt_lang_combo.show()
                
        elif input_type in ["transcription", "translation"]:
            # Hide audio file selection
            self.audio_label.hide()
            self.audio_btn.hide()
            self.audio_input.hide()
            
            # Show text file selection
            self.text_label.show()
            self.text_btn.show()
            self.text_input.show()
            
            # Hide source language selection for translation text (only for transcription text)
            if input_type == "transcription":
                self.src_lang_label.show()
                self.src_lang_combo.show()
            else:  # translation
                self.src_lang_label.hide()
                self.src_lang_combo.hide()
            
            # Show target language selection only for transcription text (for translation, we already have the text)
            if input_type == "transcription":
                self.tgt_lang_label.show()
                self.tgt_lang_combo.show()
            else:  # translation
                self.tgt_lang_label.hide()
                self.tgt_lang_combo.hide()
    
    def start_processing(self, output_mode="translation"):
        """Start the audio processing"""
        # Get input type
        input_type = self.input_type_combo.currentData()
        
        # Validate inputs
        if not self.validate_inputs(input_type, output_mode):
            return

        # Get selected target languages
        selected_target_languages = self.get_selected_target_languages()

        # Start processing in a separate thread
        self.processing_thread = ProcessingThread(
            self.audio_file_path if input_type == "audio" else None,
            self.text_file_path if input_type in ["transcription", "translation"] else None,
            self.ref_audio_path,
            self.whisper_combo.currentData(),
            self.nllb_combo.currentData(),
            self.xtts_combo.currentData(),
            self.src_lang_combo.currentData(),
            selected_target_languages,  # Pass the list of selected target languages
            input_type,
            output_mode
        )

        # Connect signals
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.status_updated.connect(self.update_status)
        self.processing_thread.log_updated.connect(self.log_message)
        self.processing_thread.processing_finished.connect(self.on_processing_finished)

        # Start the thread
        self.processing_thread.start()
    
    def get_selected_target_languages(self):
        """Get selected target language code from dropdown"""
        selected_language = self.tgt_lang_combo.currentData()
        if selected_language:
            return [selected_language]
        return []

    def validate_inputs(self, input_type="audio", output_mode="translation"):
        """Validate all input fields"""
        errors = []

        try:
            # Reference audio validation
            if not self.ref_audio_path:
                errors.append("Please select a reference audio file")
            elif not validate_audio_file(self.ref_audio_path):
                errors.append("Selected reference audio file is invalid")
            else:
                is_valid, _, msg = validate_reference_audio_duration(self.ref_audio_path)
                if not is_valid: errors.append(msg)

            # Input validation based on type
            if input_type == "audio":
                if not self.audio_file_path or not validate_audio_file(self.audio_file_path):
                    errors.append("Invalid or missing audio file")
                
                if not self.whisper_combo.currentData():
                    errors.append("Please select a Transcription model")
            else:
                if not self.text_file_path or not os.path.exists(self.text_file_path):
                    errors.append("Invalid or missing text file")

            # Model validation
            if output_mode == "translation":
                if input_type != "translation" and not ModelManager.validate_nllb_directory(self.nllb_combo.currentData()):
                    errors.append("Invalid Translation model directory")
                
                if not ModelManager.validate_xtts_directory(self.xtts_combo.currentData()):
                    errors.append("Invalid Narration model directory")

                if input_type in ["audio", "transcription"] and not self.get_selected_target_languages():
                    errors.append("Please select at least one target language")

            if errors:
                QMessageBox.critical(self, "Validation Error", "\n".join(errors))
                return False

            return True
        except Exception as e:
            logging.error(f"Validation error: {e}")
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
        self.transcript_only_btn.setEnabled(True)
        self.dubbed_translation_btn.setEnabled(True)

        if success:
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.critical(self, "Error", message)


from src.core.application.audio_orchestrator import AudioDubbingOrchestrator
from src.core.data_models.audio_models import AudioProcessingConfig, ProcessingMode


class ProcessingThread(QThread):
    """Thread for processing audio to keep UI responsive"""
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    log_updated = pyqtSignal(str)
    processing_finished = pyqtSignal(bool, str)

    def __init__(self, audio_file, text_file, ref_audio, whisper_model, nllb_model, xtts_model,
                 src_lang, tgt_lang_list, input_type, output_mode):
        super().__init__()
        self.config = AudioProcessingConfig(
            audio_file_path=audio_file,
            text_file_path=text_file,
            ref_audio_path=ref_audio,
            whisper_model_path=whisper_model,
            nllb_model_path=nllb_model,
            xtts_model_path=xtts_model,
            source_language=src_lang,
            target_languages=tgt_lang_list,
            processing_mode=ProcessingMode.TRANSCRIPTION_ONLY if output_mode == "transcript" else ProcessingMode.DUBBED_TRANSLATION
        )
        self.input_type = input_type
        self.orchestrator = AudioDubbingOrchestrator(
            progress_callback=self.progress_updated.emit,
            status_callback=self.status_updated.emit,
            log_callback=self.log_updated.emit
        )

    def stop(self):
        """Set the stop flag to terminate processing"""
        self.orchestrator.stop()

    def run(self):
        """Main processing logic"""
        success = self.orchestrator.process(self.config, self.input_type)
        if success:
            self.processing_finished.emit(True, "Processing completed successfully!")
        else:
            self.processing_finished.emit(False, "Processing failed or was stopped.")



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