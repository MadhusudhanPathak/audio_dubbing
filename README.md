# Offline Audio Dubbing

A desktop application for offline audio translation with voice cloning using Whisper, NLLB, and XTTS-v2.

## 🚀 Features

- Transcribe audio using Whisper
- Translate text using NLLB (No Language Left Behind)
- Clone voices using XTTS-v2
- Support for multiple languages
- User-friendly PyQt5 interface
- Complete offline processing (no internet required after initial setup)
- Batch processing capabilities
- Real-time progress tracking

## 📋 Prerequisites

- Python 3.8+ (Python 3.11 recommended due to PyTorch compatibility)
- At least 8GB RAM (16GB+ recommended for large models)
- Sufficient disk space for models (5-15GB depending on selected models)
- Windows, macOS, or Linux

## 🛠️ Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/offline-audio-dubbing.git
cd offline-audio-dubbing
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
   - Download from: https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-copies-downloads
   - Install both x64 and x86 versions

2. If PyTorch installation fails, try installing separately:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

## 🧰 Required Models

Before using the application, you need to download the following models:

### Whisper Models (for transcription)
- **Download from:** https://huggingface.co/ggerganov/whisper.cpp
- **Supported formats:** GGML models (.bin files)
- **Place in:** `Models/whisper/`
- **Recommended models:**
  - `ggml-tiny.bin` (~75MB, fastest but least accurate)
  - `ggml-base.bin` (~145MB, good balance)
  - `ggml-small.bin` (~465MB, more accurate)
  - `ggml-medium.bin` (~1.5GB, highly accurate)
  - `ggml-large.bin` (~2.9GB, most accurate but slower)

### NLLB Models (for translation)
- **Download from:** https://huggingface.co/facebook/nllb-200-distilled-600M or https://huggingface.co/facebook/nllb-200-3.3B
- **Place in:** `Models/nllb/`
- **Recommended models:**
  - `nllb-200-distilled-600M` (~1.2GB, good speed/accuracy)
  - `nllb-200-3.3B` (~6.6GB, highest accuracy)

### XTTS-v2 Models (for voice cloning)
- **Download from:** https://huggingface.co/coqui/XTTS-v2
- **Place in:** `Models/xtts/`
- **Recommended:** Latest version (currently 2.0.2 or newer)

## 🎯 Usage

### Starting the Application
```bash
python main.py
```

### Interface Guide
1. **Model Selection**
   - Select appropriate models from dropdown menus
   - Click "Refresh Models" to reload after downloading new models

2. **Input Files**
   - Select audio file to be processed
   - For voice cloning, provide reference audio (6-10 seconds recommended)

3. **Language Settings**
   - Select source language (or use auto-detect)
   - Choose target language for translation

4. **Processing Mode**
   - **Transcription Only:** Generate text transcript only
   - **Dubbed Translation:** Full audio translation with voice cloning

5. **Start Processing**
   - Click "Start Processing" to begin
   - Monitor progress in the progress bar and log

### Processing Workflow
1. Audio transcription using Whisper
2. Text translation using NLLB (if in dubbed mode)
3. Voice cloning and audio synthesis using XTTS-v2 (if in dubbed mode)

## 📁 Project Structure

```
offline-audio-dubbing/
├── main.py                 # Main application entry point with PyQt5 UI
├── minimal_test.py         # Lightweight UI test without heavy dependencies
├── requirements.txt        # Python dependencies
├── agent.md               # Agent configuration and architecture
├── README.md              # This file
├── Whisper.exe            # Whisper executable (Windows)
├── Whisper.dll            # Whisper dependency (Windows)
├── Inputs/                # Input audio files directory
├── Outputs/               # Generated output files directory
├── Models/                # Model storage directory
│   ├── whisper/          # Whisper model files (.bin)
│   ├── nllb/             # NLLB model directories
│   └── xtts/             # XTTS model directories
└── modules/               # Core functionality modules
    ├── transcriber.py     # Audio transcription module
    ├── translator.py      # Text translation module
    ├── voice_cloner.py    # Voice cloning and synthesis module
    └── utils.py           # Utility functions and helpers
```

## 💾 Output Format

- **Transcription Only:** `Outputs/{input_filename}_transcript.txt`
- **Full Translation:**
  - Audio: `Outputs/{input_filename}_{target_lang}.wav`
  - Transcript: `Outputs/{input_filename}_{target_lang}_transcript.txt`

## 🔧 Troubleshooting

### Common Issues and Solutions

**Q: Application crashes immediately on startup**
A: Check that all dependencies are installed. Try running `python minimal_test.py` to verify the UI works independently.

**Q: PyTorch-related errors on Windows**
A: Install Microsoft Visual C++ Redistributables or reinstall PyTorch with CPU support.

**Q: Models not appearing in dropdown**
A: Ensure models are placed in correct directories and filenames are correct. Click "Refresh Models".

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

## 🧪 Testing UI without Dependencies

If you're having trouble with heavy dependencies, test the UI with:
```bash
python minimal_test.py
```

This runs a simplified version without requiring PyTorch, Whisper, or TTS libraries.

## 📞 Support

For support, please check:
- The troubleshooting section above
- Open an issue on the GitHub repository
- Ensure all models are correctly downloaded and placed in the right directories

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Whisper models by OpenAI and whisper.cpp
- NLLB models by Meta AI
- XTTS-v2 models by Coqui AI
- PyQt5 for the GUI framework