# Offline Audio Dubbing - Agent Configuration & Context Guide

## Project Overview

This is a professional offline audio dubbing application that performs speech-to-text transcription, text translation, and voice cloning to generate dubbed audio in a target language. The application uses Whisper for transcription, NLLB for translation, and XTTS-v2 for voice cloning, all running completely offline.

## Current State

### Core Components
- **Transcriber** (`src/core/transcriber.py`): Handles audio-to-text conversion using Whisper models
- **Translator** (`src/core/translator.py`): Performs text translation using NLLB models
- **Voice Cloner** (`src/core/voice_cloner.py`): Generates dubbed audio with cloned voices using XTTS-v2
- **Helpers** (`src/utils/helpers.py`): Contains utility functions for file operations, validation, and language mapping
- **Main Application** (`src/ui/main_window.py`): PyQt5-based GUI that orchestrates the entire process
- **Configuration** (`src/config/app_config.py`): Centralized application configuration

### Architecture
```
Input Audio → Transcription (Whisper) → Translation (NLLB) → Voice Cloning (XTTS-v2) → Output Audio
```

### Recent Enhancements
1. **Professional Modular Architecture**: Clean separation of concerns with core logic, UI, utilities, and configuration modules
2. **Enhanced Error Handling**: Comprehensive input validation and error handling throughout the application
3. **Configuration Management**: Centralized configuration system with proper defaults and validation
4. **Improved Model Detection**: Enhanced model checking logic that looks for required files in both main directories and subdirectories
5. **Cleaner UI**: Removed redundant action buttons and streamlined the interface
6. **Direct Processing**: Mode selection buttons now directly trigger processing instead of requiring a separate start button
7. **Improved Layout**: Swapped positions of Browse buttons and file selection status, and reorganized language selection
8. **Concise Model Names**: Dropdowns now display only model names instead of full paths

### Key Features
- **Model Download Dialog**: Modal dialog showing required model downloads with local availability check
- **Checkbox Indicators**: Visual indication of which models are available locally
- **Refresh Capability**: Button to recheck model availability
- **Extension Information**: Clear specification of expected file extensions for each model type
- **Intuitive Processing**: Direct action buttons for transcription and dubbed translation modes
- **Comprehensive Logging**: Detailed logging with configurable levels and formats
- **Robust Error Handling**: Proper validation and error reporting at all levels

## Technical Details

### Dependencies
- PyQt5: GUI framework
- PyTorch: Neural network inference
- Transformers: NLLB model integration
- Coqui TTS: XTTS-v2 integration
- SoundFile, PyDub: Audio processing
- Various utility libraries

### Professional File Structure
```
Offline Audio Dubbing/
├── main.py                 # Main application entry point
├── requirements.txt        # Python dependencies
├── agent.md               # This file
├── README.md              # Project documentation
├── LICENSE                # License information
├── Whisper.exe            # Whisper executable (Windows)
├── Whisper.dll            # Whisper dependency (Windows)
├── Inputs/                # Input audio files directory
├── Outputs/               # Generated output files directory
├── Models/                # Model storage directory
│   ├── whisper/          # Whisper model files (.bin or .gguf)
│   ├── nllb/             # NLLB model directories/files
│   └── xtts/             # XTTS model directories/files
├── src/                   # Source code root
│   ├── __init__.py       # Package initialization
│   ├── core/             # Core business logic
│   │   ├── __init__.py
│   │   ├── transcriber.py # Audio transcription module
│   │   ├── translator.py  # Text translation module
│   │   └── voice_cloner.py # Voice cloning module
│   ├── ui/               # User interface components
│   │   ├── __init__.py
│   │   └── main_window.py # Main UI window
│   ├── utils/            # Utility functions
│   │   ├── __init__.py
│   │   └── helpers.py    # Helper functions
│   ├── config/           # Configuration module
│   │   ├── __init__.py
│   │   └── app_config.py # Application configuration
│   └── models/           # Data models (if any)
├── docs/                 # Documentation files
├── tests/                # Unit and integration tests
├── config/               # Configuration files
└── scripts/              # Utility scripts
```

### Processing Workflow
1. User selects models, input audio, reference audio, and languages
2. Application validates all inputs and file paths
3. Transcription phase: Audio is converted to text using Whisper
4. Translation phase: Text is translated using NLLB (if in dubbed mode)
5. Voice cloning phase: Translated text is synthesized with cloned voice using XTTS-v2
6. Output is saved to the Outputs directory

### Error Handling Strategy
- All file operations include existence checks
- Model loading includes validation of model files
- Audio processing includes format validation
- Each processing step has specific error handling
- User receives informative error messages
- All errors are logged for debugging
- Comprehensive input validation at all levels

### Logging Configuration
- Logs are written to both file (`offline_dubbing.log`) and console
- Different log levels (INFO, WARNING, ERROR, DEBUG) are used appropriately
- All major operations and errors are logged with timestamps
- Model loading, processing steps, and errors are tracked

## Context for Further Development

### Key Areas for Potential Improvement
1. **Performance Optimization**: Model loading and processing speed
2. **Memory Management**: For handling large audio files
3. **Batch Processing**: Support for processing multiple files
4. **Advanced Audio Processing**: More sophisticated audio format handling
5. **Model Management**: Better model versioning and updates
6. **UI Enhancements**: Additional user experience improvements
7. **Testing Coverage**: Comprehensive unit and integration tests
8. **Documentation**: API documentation and usage examples

### Integration Points
- The application is modular and each component can be updated independently
- The GUI communicates with processing modules through a well-defined interface
- Logging system can be extended for more detailed analytics
- Error handling can be enhanced with custom exception types
- Configuration system allows for easy customization

### Common Issues and Solutions
- **Model Loading Failures**: Check model paths and file integrity
- **Audio Format Issues**: Ensure audio files are in supported formats
- **Memory Issues**: Use smaller models or process shorter audio segments
- **Dependency Conflicts**: Use virtual environments and specific versions
- **Version Compatibility**: Ensure PyTorch and other dependencies are compatible

## Usage Patterns

### Normal Operation Flow
1. Launch main.py
2. If models are missing, model download dialog appears with availability check
3. If all models are present, skip directly to main interface
4. Select appropriate models from dropdowns (model names only)
5. Choose input audio file
6. For voice cloning, provide reference audio (6-10 seconds)
7. Select source and target languages
8. Choose processing mode (transcription only or dubbed translation) - clicking either button starts processing directly
9. Monitor progress in the progress bar and log

### Error Recovery
- Invalid inputs are caught and reported to user
- Processing can be stopped with internal stop mechanism
- Errors are logged for debugging
- Application remains responsive during processing

This configuration provides a robust foundation for offline audio dubbing with comprehensive error handling, logging, and validation mechanisms, plus enhanced user experience features. The professional modular architecture enables easy maintenance and future enhancements.