# Offline Audio Dubbing

A professional desktop application for offline audio translation with voice cloning using OpenMOSS MOSS-Audio, NLLB, and OpenMOSS MOSS-TTS.

## 🚀 Features

- Transcribe audio using OpenMOSS MOSS-Audio
- Translate text using NLLB (No Language Left Behind)
- Clone voices using OpenMOSS MOSS-TTS
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

The application detects your hardware (CUDA/VRAM/RAM) on startup and tells you exactly which model tier to download and where to place it — see the model info dialog shown on first launch, or run the app to have it create the required `Models/` subdirectories automatically. In general:

### Transcription Models (OpenMOSS MOSS-Audio)

- **Download from:** <https://huggingface.co/OpenMOSS-Team> (MOSS-Audio-4B-Instruct or MOSS-Audio-8B-Instruct, depending on your hardware tier)
- **Expected:** a model directory containing `config.json` and model weights
- **Place in:** `Models/moss-audio/`

### Translation Models (NLLB)

- **Download from:** <https://huggingface.co/facebook/nllb-200-distilled-600M> or <https://huggingface.co/facebook/nllb-200-3.3B>
- **Required files:** config.json, pytorch_model.bin, tokenizer.json, generation_config.json
- **Place in:** `Models/nllb/` (either directly in the folder or in a subdirectory)
- **Recommended models:**
  - `nllb-200-distilled-600M` (~1.2GB, good speed/accuracy)
  - `nllb-200-3.3B` (~6.6GB, highest accuracy)

### Narration Models (OpenMOSS MOSS-TTS)

- **Download from:** <https://huggingface.co/OpenMOSS-Team> (MOSS-TTS, MOSS-TTS-Local-Transformer, MOSS-TTS-Nano, or MOSS-TTS-GGUF, depending on your hardware tier)
- **Expected:** a model directory containing `config.json` and model weights
- **Place in:** `Models/moss-tts/`

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
   - Uses MOSS-Audio for speech-to-text

2. **Full Audio Dubbing**
   - Input: Audio file + reference voice audio
   - Output: Translated and dubbed audio files in multiple languages
   - Uses: MOSS-Audio → NLLB → MOSS-TTS pipeline

3. **Text Translation & Dubbing**
   - Input: Transcription text file + reference voice audio
   - Output: Translated text and dubbed audio files
   - Uses: NLLB → MOSS-TTS pipeline

4. **Direct Voice Synthesis**
   - Input: Translation text file + reference voice audio
   - Output: Dubbed audio file
   - Uses: MOSS-TTS only

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
├── AGENTS.md                            # Agent configuration and architecture
├── README.md                            # This file
├── LICENSE                              # License information
├── Inputs/                              # Input audio files directory
├── Outputs/                             # Generated output files directory
├── Models/                              # Model storage directory
│   ├── moss-audio/                      # MOSS-Audio (STT) model files
│   ├── nllb/                            # NLLB model directories/files
│   └── moss-tts/                        # MOSS-TTS model files
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
│   │       ├── base_model_service.py    # Shared STT/TTS service lifecycle
│   │       ├── transcription_service.py # MOSS-Audio integration
│   │       ├── translation_service.py   # NLLB integration
│   │       ├── voice_synthesis_service.py # MOSS-TTS integration
│   │       └── model_manager.py         # Local model scanning
│   ├── api/                             # User interface layer
│   │   ├── __init__.py
│   │   └── interfaces/
│   │       ├── __init__.py
│   │       ├── gui_interface.py         # Main PyQt5 GUI
│   │       └── dialogs.py               # Dialog components
│   └── utils/                           # Utility functions
│       ├── model_setup_checker.py       # Hardware detection & model tier selection
│       └── common/
│           ├── __init__.py
│           ├── helpers.py               # Utilities, language support, validation
│           └── app_config.py            # Configuration constants
└── tests/                               # Manual verification scripts
```

## 🏗️ Architecture Overview

### Technology Stack
- **UI Framework**: PyQt5 (cross-platform desktop interface)
- **Speech Recognition**: OpenMOSS MOSS-Audio (offline, local HuggingFace model)
- **Machine Translation**: NLLB-200 (multilingual, state-of-the-art)
- **Voice Synthesis**: OpenMOSS MOSS-TTS (voice cloning, multilingual)
- **Deep Learning**: PyTorch with GPU support
- **Audio Processing**: soundfile, pydub, torchaudio

### Core Components

| Component | File | Purpose | Key Features |
|-----------|------|---------|--------------|
| **Transcriber** | `src/core/services/transcription_service.py` | Speech-to-text | MOSS-Audio integration, optional language hint, timestamp mode |
| **Translator** | `src/core/services/translation_service.py` | Text translation | NLLB-200 support, quantization, memory optimization |
| **VoiceSynthesizer** | `src/core/services/voice_synthesis_service.py` | Voice synthesis | MOSS-TTS integration, voice cloning, multilingual support |
| **AudioOrchestrator** | `src/core/application/audio_orchestrator.py` | Workflow management | Pipeline coordination, error handling, results aggregation |
| **MainWindow** | `src/api/interfaces/gui_interface.py` | User interface | PyQt5 GUI, model selection, real-time progress, logging |
| **Dialogs** | `src/api/interfaces/dialogs.py` | Dialog windows | Model info, status dialogs, check model availability |
| **Helpers** | `src/utils/common/helpers.py` | Utilities | Language mapping, audio validation, file operations |
| **Configuration** | `src/utils/common/app_config.py` | Settings | Paths, formats, timeouts, logging configuration |
| **Data Models** | `src/core/data_models/audio_models.py` | Type safety | Dataclasses for configs, type hints |

### Processing Pipeline

```
Audio Input
    ↓ (MOSS-Audio)
Text Transcription
    ↓ (NLLB)
Translated Text
    ↓ (MOSS-TTS)
Dubbed Audio Output
```

## 🔧 Configuration

All configuration is centralized in `src/utils/common/app_config.py`:

- **Model Paths**: MOSS-Audio, NLLB, MOSS-TTS directory locations
- **Audio Settings**: Supported formats, min/max reference audio duration
- **Processing**: Transcription timeout
- **Logging**: Log level, format, output files

Hardware detection (CPU/CUDA, VRAM tier, and which MOSS-Audio/MOSS-TTS variant to use) is handled separately at startup by `src/utils/model_setup_checker.py`.

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
   - Use the 4B MOSS-Audio/MOSS-TTS tier for speed on smaller GPUs
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