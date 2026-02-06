# Quick Start Guide - Offline Audio Dubbing

## Overview
This guide will help you get started with the Offline Audio Dubbing application quickly. Follow these steps to set up and run the application.

## Prerequisites
- Python 3.8 or higher (Python 3.11 recommended)
- At least 4GB of free disk space for models
- 8GB RAM recommended

## Step 1: Install Dependencies

1. Open a terminal/command prompt
2. Navigate to the project directory
3. Run the following commands:

```bash
# Create a virtual environment (recommended)
python -m venv venv
# Activate it (Windows)
venv\Scripts\activate
# Activate it (macOS/Linux)
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Download Models

The application requires three types of models. Download them from the following sources:

### Whisper Models (for transcription)
- Go to: https://huggingface.co/ggerganov/whisper.cpp
- Download a model file (e.g., `ggml-base.bin`)
- Place it in the `Models/whisper/` directory

### NLLB Models (for translation)
- Go to: https://huggingface.co/facebook/nllb-200-distilled-600M
- Download the model files
- Extract to a folder in `Models/nllb/` directory

### XTTS-v2 Models (for voice cloning)
- Go to: https://huggingface.co/coqui/XTTS-v2
- Download the model files
- Extract to a folder in `Models/xtts/` directory

## Step 3: Run the Application

1. Make sure you're in the project directory
2. Activate your virtual environment if you created one
3. Run the application:

```bash
python main.py
```

## Step 4: Using the Application

### Interface Overview
1. **Model Selection**: Choose your models from the dropdown menus
2. **Input Files**: Select your audio file and reference audio (for voice cloning)
3. **Language Settings**: Select source and target languages
4. **Processing Mode**: Choose between "Transcription Only" or "Dubbed Translation"
5. **Start Processing**: Click the "Start Processing" button

### Basic Workflow
1. Select your models from the dropdown menus
2. Click "Refresh Models" if you just added new models
3. Browse and select your input audio file
4. For voice cloning, browse and select a reference audio file (6-10 seconds recommended)
5. Select source language (or use "Auto-detect")
6. Select target language for translation
7. Choose processing mode:
   - **Transcription Only**: Creates a text transcript
   - **Dubbed Translation**: Creates translated audio with cloned voice
8. Click "Start Processing" and monitor progress

## Example Use Cases

### Transcription Only
1. Select "Transcription Only" mode
2. Choose an audio file
3. Select source language or use auto-detect
4. Click "Start Processing"
5. Find the transcript in the `Outputs/` directory

### Dubbed Translation
1. Select "Dubbed Translation" mode
2. Choose an audio file
3. Provide a reference audio file for voice cloning
4. Select source and target languages
5. Click "Start Processing"
6. Find the dubbed audio in the `Outputs/` directory

## Troubleshooting Quick Fixes

### Application Won't Start
- Ensure all dependencies are installed
- Check that Python 3.8+ is installed
- Verify the virtual environment is activated

### Models Don't Appear in Dropdown
- Verify models are in the correct directories
- Click "Refresh Models" button
- Check file extensions are correct

### Audio Format Not Supported
- Convert to WAV, MP3, FLAC, M4A, AAC, OGG, or WMA
- Use audio conversion tools if needed

### Processing Fails
- Check the log area for specific error messages
- Ensure sufficient disk space
- Try with a smaller model for faster processing

## Next Steps

- Experiment with different model sizes for speed vs. quality trade-offs
- Try different reference audio samples for better voice cloning results
- Explore batch processing for multiple files
- Check the full documentation for advanced features