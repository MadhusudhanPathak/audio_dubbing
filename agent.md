# Offline Audio Dubbing - Agent Configuration & Context Guide

## Project Overview

This is an offline audio dubbing application that performs speech-to-text transcription, text translation, and voice cloning to generate dubbed audio in a target language. The application uses Whisper for transcription, NLLB for translation, and XTTS-v2 for voice cloning, all running completely offline.

## Current State

### Core Components
- **Transcriber** (`modules/transcriber.py`): Handles audio-to-text conversion using Whisper models
- **Translator** (`modules/translator.py`): Performs text translation using NLLB models
- **Voice Cloner** (`modules/voice_cloner.py`): Generates dubbed audio with cloned voices using XTTS-v2
- **Utils** (`modules/utils.py`): Contains utility functions for file operations, validation, and language mapping
- **Main Application** (`main.py`): PyQt5-based GUI that orchestrates the entire process

### Architecture
```
Input Audio → Transcription (Whisper) → Translation (NLLB) → Voice Cloning (XTTS-v2) → Output Audio
```

### Robustness Features Added
1. **Comprehensive Error Handling**: Each module has specific exception handling for different error types (FileNotFoundError, ValueError, RuntimeError)
2. **Detailed Logging System**: Full logging with file output (`offline_dubbing.log`) and console output
3. **Input Validation**: All file paths and inputs are validated before processing
4. **Resource Management**: Proper cleanup methods and resource handling
5. **Progress Tracking**: Granular progress updates and status messages
6. **Filename Sanitization**: Prevents issues with special characters in output filenames

### Key Enhancements
- **Enhanced Whisper Parsing**: Better output parsing and language detection from Whisper.exe
- **Model Validation**: Checks for model existence before loading
- **Audio Validation**: Validates audio files before processing
- **Directory Management**: Ensures output directories exist before file operations
- **Graceful Degradation**: Handles missing models and files gracefully with informative messages

## Technical Details

### Dependencies
- PyQt5: GUI framework
- PyTorch: Neural network inference
- Transformers: NLLB model integration
- Coqui TTS: XTTS-v2 integration
- SoundFile, PyDub: Audio processing
- Various utility libraries

### File Structure
```
Offline Audio Dubbing/
├── main.py                 # Main application with PyQt5 GUI
├── requirements.txt        # Python dependencies
├── agent.md               # This file
├── README.md              # Project documentation
├── Whisper.exe            # Whisper executable (Windows)
├── Whisper.dll            # Whisper dependency (Windows)
├── Inputs/                # Input audio files directory
├── Outputs/               # Generated output files directory
├── Models/                # Model storage directory
│   ├── whisper/          # Whisper model files (.bin)
│   ├── nllb/             # NLLB model directories
│   └── xtts/             # XTTS model directories
├── modules/               # Core functionality modules
│   ├── transcriber.py     # Audio transcription module
│   ├── translator.py      # Text translation module
│   ├── voice_cloner.py    # Voice cloning and synthesis module
│   └── utils.py           # Utility functions and helpers
└── docs/                  # Documentation files
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

### Integration Points
- The application is modular and each component can be updated independently
- The GUI communicates with processing modules through a well-defined interface
- Logging system can be extended for more detailed analytics
- Error handling can be enhanced with custom exception types

### Common Issues and Solutions
- **Model Loading Failures**: Check model paths and file integrity
- **Audio Format Issues**: Ensure audio files are in supported formats
- **Memory Issues**: Use smaller models or process shorter audio segments
- **Dependency Conflicts**: Use virtual environments and specific versions

## Usage Patterns

### Normal Operation Flow
1. Launch main.py
2. Select appropriate models from dropdowns
3. Choose input audio file
4. For voice cloning, provide reference audio (6-10 seconds)
5. Select source and target languages
6. Choose processing mode (transcription only or dubbed translation)
7. Start processing and monitor progress

### Error Recovery
- Invalid inputs are caught and reported to user
- Processing can be stopped with the Stop button
- Errors are logged for debugging
- Application remains responsive during processing

This configuration provides a robust foundation for offline audio dubbing with comprehensive error handling, logging, and validation mechanisms.