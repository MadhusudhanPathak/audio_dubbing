# Offline Audio Dubbing

A professional desktop application for offline audio translation with voice cloning using Whisper, NLLB, and XTTS-v2.

## 🚀 Features

- Transcribe audio using Whisper
- Translate text using NLLB (No Language Left Behind)
- Clone voices using XTTS-v2
- Support for multiple languages
- User-friendly PyQt5 interface
- Complete offline processing (no internet required after initial setup)
- Real-time progress tracking
- Smart model availability checking with automatic dialog skipping when all models are present
- Intuitive processing mode selection with direct action buttons
- Professional modular architecture with clean separation of concerns
- Comprehensive error handling and logging
- Type-safe implementations with proper validation
- Clean separation of business logic, UI, and utilities
- Improved application orchestration with dedicated workflow management
- Enhanced data models for better structure and maintainability
- Modern directory structure following industry standards

## 📋 Prerequisites

- Python 3.8+ (Python 3.11 recommended due to PyTorch compatibility)
- At least 8GB RAM (16GB+ recommended for large models)
- Sufficient disk space for models (5-15GB depending on selected models)
- Windows, macOS, or Linux

## 🛠️ Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/MadhusudhanPathak/audio_dubbing
cd audio_dubbing
```

### Step 2: Set Up Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### ⚠️ Windows-Specific Installation Notes

If you encounter DLL errors when running the application on Windows:

1. Install Microsoft Visual C++ Redistributable for Visual Studio:
   - Download from: https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-copies-downloads
   - Install both x64 and x86 versions

2. If PyTorch installation fails, try installing separately:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch>=2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

## 🧰 Required Models

Before using the application, you need to download the following models:

### Transcription Models (Whisper)
- **Download from:** https://huggingface.co/ggerganov/whisper.cpp
- **Supported formats:** GGML models (.bin files) or GGUF format (.gguf files)
- **Place in:** `Models/whisper/`
- **Recommended models:**
  - `ggml-tiny.bin` (~75MB, fastest but least accurate)
  - `ggml-base.bin` (~145MB, good balance)
  - `ggml-small.bin` (~465MB, more accurate)
  - `ggml-medium.bin` (~1.5GB, highly accurate)
  - `ggml-large.bin` (~2.9GB, most accurate but slower)

### Translation Models (NLLB)
- **Download from:** https://huggingface.co/facebook/nllb-200-distilled-600M or https://huggingface.co/facebook/nllb-200-3.3B
- **Required files:** config.json, pytorch_model.bin, tokenizer.json, generation_config.json
- **Place in:** `Models/nllb/` (either directly in the folder or in a subdirectory)
- **Recommended models:**
  - `nllb-200-distilled-600M` (~1.2GB, good speed/accuracy)
  - `nllb-200-3.3B` (~6.6GB, highest accuracy)

### Narration Models (XTTS-v2)
- **Download from:** https://huggingface.co/coqui/XTTS-v2
- **Required files:** config.json, model.pth, vocab.json (for newer versions)
- **Place in:** `Models/xtts/` (either directly in the folder or in a subdirectory)
- **Recommended:** Latest version (currently 2.0.2 or newer)

## 🎯 Usage

### Starting the Application
```bash
python main.py
```

### Application Workflow

The application supports multiple processing workflows:

1. **Audio Transcription Only**
   - Input: Audio file
   - Output: Transcription text file
   - Uses Whisper for speech-to-text

2. **Full Audio Dubbing**
   - Input: Audio file + reference voice audio
   - Output: Translated and dubbed audio files in multiple languages
   - Uses: Whisper → NLLB → XTTS-v2 pipeline

3. **Text Translation & Dubbing**
   - Input: Transcription text file + reference voice audio
   - Output: Translated text and dubbed audio files
   - Uses: NLLB → XTTS-v2 pipeline

4. **Direct Voice Synthesis**
   - Input: Translation text file + reference voice audio
   - Output: Dubbed audio file
   - Uses: XTTS-v2 only

### Step-by-Step Guide

1. **Launch Application**: Run `python main.py`
2. **Check Models**: Application verifies model availability on startup
3. **Download Models** (if needed): Follow links in the model info dialog
4. **Select Input**: Choose audio file or text file based on your workflow
5. **Configure Options**:
   - Source language (or Auto-detect for audio)
   - Target language(s)
   - Reference audio for voice cloning (for dubbed output)
6. **Select Models**: Choose appropriate models for each component
7. **Process**: Click "Process" and monitor progress in real-time
8. **Review Output**: Check the `Outputs/` folder for results

## 📁 Professional Project Structure

```
offline-audio-dubbing/
├── main.py                              # Main application entry point
├── requirements.txt                     # Python dependencies
├── agent.md                             # Agent configuration and architecture
├── README.md                            # This file
├── LICENSE                              # License information
├── Whisper.exe                          # Whisper executable (Windows)
├── Whisper.dll                          # Whisper dependency (Windows)
├── Inputs/                              # Input audio files directory
├── Outputs/                             # Generated output files directory
├── Models/                              # Model storage directory
│   ├── whisper/                         # Whisper model files (.bin or .gguf)
│   ├── nllb/                            # NLLB model directories/files
│   └── xtts/                            # XTTS model directories/files
├── src/                                 # Source code root
│   ├── __init__.py                      # Package initialization
│   ├── core/                            # Core application logic
│   │   ├── __init__.py
│   │   ├── application/
│   │   │   ├── __init__.py
│   │   │   └── audio_orchestrator.py    # Workflow orchestration
│   │   ├── data_models/
│   │   │   ├── __init__.py
│   │   │   └── audio_models.py          # Type-safe data structures
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── transcription_service.py # Whisper integration
│   │       ├── translation_service.py   # NLLB integration
│   │       └── voice_synthesis_service.py # XTTS integration
│   ├── api/                             # User interface layer
│   │   ├── __init__.py
│   │   └── interfaces/
│   │       ├── __init__.py
│   │       ├── gui_interface.py         # Main PyQt5 GUI
│   │       └── dialogs.py               # Dialog components
│   └── utils/                           # Utility functions
│       └── common/
│           ├── __init__.py
│           ├── helpers.py               # Utilities, language support, validation
│           └── app_config.py            # Configuration constants
├── config/                              # Configuration files
├── docs/                                # Documentation
├── scripts/                             # Build and utility scripts
└── tests/                               # Test suite (future expansion)
```

## 🏗️ Architecture Overview

### Technology Stack
- **UI Framework**: PyQt5 (cross-platform desktop interface)
- **Speech Recognition**: Whisper (offline, supports 99+ languages)
- **Machine Translation**: NLLB-200 (205+ languages, state-of-the-art)
- **Voice Synthesis**: XTTS-v2 (voice cloning, multilingual)
- **Deep Learning**: PyTorch with GPU support
- **Audio Processing**: librosa, soundfile, pydub

### Core Components

| Component | File | Purpose | Key Features |
|-----------|------|---------|--------------|
| **Transcriber** | `src/core/services/transcription_service.py` | Speech-to-text | Whisper.cpp integration, language detection, timeout handling |
| **Translator** | `src/core/services/translation_service.py` | Text translation | NLLB-200 support, quantization, 205+ languages, memory optimization |
| **VoiceCloner** | `src/core/services/voice_synthesis_service.py` | Voice synthesis | XTTS-v2 integration, voice cloning, multilingual support |
| **AudioOrchestrator** | `src/core/application/audio_orchestrator.py` | Workflow management | Pipeline coordination, error handling, results aggregation |
| **MainWindow** | `src/api/interfaces/gui_interface.py` | User interface | PyQt5 GUI, model selection, real-time progress, logging |
| **Dialogs** | `src/api/interfaces/dialogs.py` | Dialog windows | Model info, status dialogs, check model availability |
| **Helpers** | `src/utils/common/helpers.py` | Utilities | 205 languages mapping, audio validation, file operations |
| **Configuration** | `src/utils/common/app_config.py` | Settings | Paths, formats, timeouts, logging configuration |
| **Data Models** | `src/core/data_models/audio_models.py` | Type safety | Dataclasses for configs, results, type hints |

### Processing Pipeline

```
Audio Input
    ↓ (Whisper)
Text Transcription
    ↓ (NLLB)
Translated Text
    ↓ (XTTS-v2)
Dubbed Audio Output
```

## 🔧 Configuration

All configuration is centralized in `src/utils/common/app_config.py`:

- **Model Paths**: Whisper, NLLB, XTTS directory locations
- **Audio Settings**: Supported formats, min/max reference audio duration
- **Processing**: Transcription timeout, default device (CPU/CUDA)
- **Logging**: Log level, format, output files

## 🚀 Advanced Usage

### GPU Support
The application automatically detects and uses GPU (CUDA) when available:
```python
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

### Quantization
The NLLB translator supports 8-bit quantization for reduced memory usage:
```python
translator = Translator(model_path, use_quantization=True)
```

### Batch Processing
Process multiple files by running the application multiple times or implementing batch scripts.

## 🐛 Troubleshooting

### Common Issues

**Issue**: "Whisper.exe not found"
- **Solution**: Download from https://github.com/ggerganov/whisper.cpp/releases and place in project root

**Issue**: "CUDA out of memory"
- **Solutions**:
  - Use 8-bit quantization: `use_quantization=True`
  - Use a smaller reference audio
  - Reduce input text length
  - Use CPU instead of GPU

**Issue**: "Model configuration mismatch"
- **Solution**: Ensure models match expected format (config.json, pytorch_model.bin for NLLB)

**Issue**: Poor dubbing quality
- **Solutions**:
  - Use reference audio between 6-30 seconds
  - Ensure reference audio matches target language
  - Try different reference audio files

## 📊 Performance Tips

1. **Model Selection**:
   - Use smaller Whisper models for speed (ggml-tiny, ggml-base)
   - Use NLLB-200-distilled-600M for speed, 3.3B for accuracy

2. **Audio Setup**:
   - Reference audio: 6-15 seconds is optimal
   - Target temperature: Keep audio sample rate at 22kHz or 44kHz

3. **Memory Management**:
   - Enable quantization for models with limited VRAM
   - Process one language at a time for large batches

4. **Threading**:
   - Processing runs in separate thread to maintain UI responsiveness
   - Multiple models can be loaded simultaneously (with available RAM)

## 📝 Type Safety & Error Handling

The codebase includes:
- Comprehensive type hints throughout
- Custom exception classes for specific error cases
- Input validation for all user-facing functions
- Detailed error messages and logging

## 🧪 Testing

Currently, the application includes:
- Integration testing via GUI
- Model availability checking
- File validation on load
- Configuration validation

Future expansions can add:
- Unit tests for individual services
- Integration tests for workflows
- Performance benchmarking
- Stress testing with various file sizes

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

### Interface Guide
1. **Model Selection**
   - The application automatically checks for model availability
   - If all models are present, the model download dialog is skipped
   - Select appropriate models from dropdown menus (only model names are displayed)
   - Click "Refresh Models" to reload after downloading new models

2. **Input Files**
   - Select audio file to be processed using the Browse button
   - For voice cloning, provide reference audio (6-10 seconds recommended)

3. **Language Settings**
   - Select source language (or use auto-detect)
   - Choose target language for translation

4. **Processing Mode**
   - **Transcription Only:** Generate text transcript only
   - **Dubbed Translation:** Full audio translation with voice cloning
   - Clicking either button directly starts the corresponding process

### Processing Workflow
1. Audio transcription using Whisper
2. Text translation using NLLB (if in dubbed mode)
3. Voice cloning and audio synthesis using XTTS-v2 (if in dubbed mode)

## 📁 Professional Project Structure

```
offline-audio-dubbing/
├── main.py                 # Main application entry point
├── requirements.txt        # Python dependencies
├── agent.md               # Agent configuration and architecture
├── README.md              # This file
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
│   ├── core/             # Core application logic
│   │   ├── __init__.py
│   │   ├── application/  # Application orchestration layer
│   │   │   ├── __init__.py
│   │   │   └── audio_orchestrator.py # Workflow management
│   │   ├── data_models/  # Data structures and models
│   │   │   ├── __init__.py
│   │   │   └── audio_models.py # Data classes for audio processing
│   │   └── services/     # Business logic and service implementations
│   │       ├── __init__.py
│   │       ├── transcription_service.py # Audio transcription service
│   │       ├── translation_service.py   # Text translation service
│   │       └── voice_synthesis_service.py # Voice synthesis service
│   ├── api/              # User interfaces and API endpoints
│   │   ├── __init__.py
│   │   └── interfaces/   # GUI and API interfaces
│   │       ├── __init__.py
│   │       └── gui_interface.py # Main GUI interface
│   └── utils/            # Shared utilities and configuration
│       ├── __init__.py
│       └── common/       # Helper functions and configuration
│           ├── __init__.py
│           ├── helpers.py    # Helper functions
│           └── app_config.py # Application configuration
└── tests/                # Unit and integration tests
```

## 💾 Output Format

- **Transcription Only:** `Outputs/{input_filename}_transcript_{timestamp}.txt`
- **Translation Text:** `Outputs/{input_filename}_translation_{language}_{timestamp}.txt`
- **Dubbed Audio:** `Outputs/{input_filename}_dubbed_{language}_{timestamp}.wav`

## 🔧 Troubleshooting

### Common Issues and Solutions

**Q: Model download dialog keeps appearing even when models are present**
A: Ensure all required model files are present. The application checks for specific files in each model directory.

**Q: Models not appearing in dropdown**
A: Ensure models are placed in correct directories with required files. Click "Refresh Models".

**Q: Audio format not supported**
A: Convert to supported formats: WAV, MP3, FLAC, M4A, AAC, OGG, WMA.

**Q: Out of memory errors**
A: Use smaller models or increase system virtual memory. Close other applications.

**Q: Slow processing**
A: Use smaller models for faster processing. Consider using GPU if available.

### Performance Tips
- Use smaller Whisper models for faster transcription
- Use distilled NLLB models for faster translation
- Ensure sufficient RAM for model loading
- Process shorter audio segments for faster results

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Whisper models by OpenAI and whisper.cpp
- NLLB models by Meta AI
- XTTS-v2 models by Coqui AI
- PyQt5 for the GUI framework