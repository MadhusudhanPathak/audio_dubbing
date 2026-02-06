# Offline Audio Dubbing - Technical Documentation

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Module Descriptions](#module-descriptions)
3. [API Reference](#api-reference)
4. [Configuration Guide](#configuration-guide)
5. [Troubleshooting](#troubleshooting)

## Architecture Overview

The Offline Audio Dubbing application follows a modular architecture with clear separation of concerns. The system processes audio through three main stages: transcription, translation, and voice cloning.

```
[Input Audio] → [Transcriber] → [Translator] → [Voice Cloner] → [Output Audio]
```

Each stage is handled by a dedicated module that can operate independently or as part of the full pipeline.

## Module Descriptions

### 1. Transcriber Module (`modules/transcriber.py`)

The transcriber module handles speech-to-text conversion using Whisper models.

#### Key Features:
- Automatic language detection
- High-accuracy transcription
- Support for multiple Whisper model sizes
- Subprocess integration with Whisper.exe

#### Class: `Transcriber`
```python
class Transcriber:
    def __init__(self, model_path):
        """Initialize the transcriber with a specific model."""
    
    def transcribe(self, audio_file, language=None):
        """Transcribe audio file to text.
        
        Args:
            audio_file (str): Path to the audio file
            language (str, optional): Language code, auto-detect if None
            
        Returns:
            dict: Contains 'text' and 'language' keys
        """
```

#### Implementation Details:
- Uses Whisper.cpp executable for efficient processing
- Supports GGML model format for CPU optimization
- Handles various audio formats through automatic conversion
- Implements error handling for corrupted audio files

### 2. Translator Module (`modules/translator.py`)

The translator module converts text between languages using NLLB models.

#### Key Features:
- Support for 200+ languages
- High-fidelity translation preserving context
- Integration with Hugging Face Transformers
- Efficient batch processing

#### Class: `Translator`
```python
class Translator:
    def __init__(self, model_path):
        """Initialize the translator with a specific model."""
    
    def translate(self, text, source_lang, target_lang):
        """Translate text from source to target language.
        
        Args:
            text (str): Text to translate
            source_lang (str): Source language code
            target_lang (str): Target language code
            
        Returns:
            str: Translated text
        """
```

#### Implementation Details:
- Leverages NLLB (No Language Left Behind) models
- Handles language code mapping between different systems
- Implements context-aware translation for better quality
- Includes preprocessing for optimal translation results

### 3. Voice Cloner Module (`modules/voice_cloner.py`)

The voice cloner generates synthetic speech with cloned voice characteristics.

#### Key Features:
- Voice cloning from reference audio
- Natural speech synthesis
- Support for multiple speakers
- High-quality audio output

#### Class: `VoiceCloner`
```python
class VoiceCloner:
    def __init__(self, model_path):
        """Initialize the voice cloner with a specific model."""
    
    def clone_voice(self, text, reference_audio, output_path, language):
        """Generate audio with cloned voice.
        
        Args:
            text (str): Text to synthesize
            reference_audio (str): Path to reference audio for voice cloning
            output_path (str): Path for output audio file
            language (str): Language code for synthesis
        """
```

#### Implementation Details:
- Uses XTTS-v2 for state-of-the-art voice cloning
- Preserves speaker characteristics from reference audio
- Generates high-quality audio in WAV format
- Implements prosody transfer for natural-sounding speech

### 4. Utilities Module (`modules/utils.py`)

The utilities module provides helper functions for file operations and validation.

#### Key Functions:
- Model file scanning
- Audio format validation
- Directory management
- Language code mapping

#### Key Functions:
```python
def scan_model_files(directory, extension):
    """Scan directory for model files with specified extension."""
    
def validate_audio_file(file_path):
    """Validate if audio file is in a supported format."""
    
def validate_reference_audio_duration(file_path):
    """Validate reference audio duration is appropriate for voice cloning."""
    
def map_language_code(code, target_format):
    """Convert language codes between different formats."""
    
def ensure_directory_exists(path):
    """Ensure directory exists, creating if necessary."""
    
def get_supported_audio_formats():
    """Get list of supported audio formats."""
```

## API Reference

### Main Application Flow

The main application orchestrates the modules through the `ProcessingThread` class:

```python
class ProcessingThread(QThread):
    # Signals for UI updates
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    log_updated = pyqtSignal(str)
    processing_finished = pyqtSignal(bool, str)
    
    def run(self):
        """Execute the full processing pipeline."""
        # 1. Initialize transcriber
        transcriber = Transcriber(self.whisper_model)
        
        # 2. Perform transcription
        transcription_result = transcriber.transcribe(self.audio_file, self.src_lang)
        
        # 3. Handle transcription-only mode
        if self.transcription_only:
            # Save transcription and exit
            return
        
        # 4. Initialize translator
        translator = Translator(self.nllb_model)
        
        # 5. Perform translation
        translated_text = translator.translate(
            transcription_result["text"],
            transcription_result["language"],
            self.tgt_lang
        )
        
        # 6. Initialize voice cloner
        cloner = VoiceCloner(self.xtts_model)
        
        # 7. Generate dubbed audio
        cloner.clone_voice(
            translated_text,
            self.ref_audio,
            output_path,
            self.tgt_lang.split('_')[0]
        )
```

### Language Support

The application supports multiple language codes in different formats:

| Format | Example | Description |
|--------|---------|-------------|
| NLLB | eng_Latn | NLLB language codes with script |
| ISO 639-1 | en | Two-letter language codes |
| Full Name | English | Human-readable language names |

## Configuration Guide

### Model Directory Structure

The application expects models in the following directory structure:

```
Models/
├── whisper/
│   ├── ggml-tiny.bin
│   ├── ggml-base.bin
│   └── ...
├── nllb/
│   └── nllb-200-distilled-600M/
│       ├── config.json
│       ├── pytorch_model.bin
│       └── ...
└── xtts/
    └── v2.0.2/
        ├── config.json
        ├── model.pth
        └── ...
```

### Audio File Requirements

#### Input Audio
- **Formats:** WAV, MP3, FLAC, M4A, AAC, OGG, WMA
- **Sample Rate:** 16kHz recommended (automatic conversion available)
- **Channels:** Mono or stereo
- **Duration:** No strict limits, but longer files take more processing time

#### Reference Audio (for voice cloning)
- **Duration:** 6-10 seconds recommended
- **Quality:** Clear, high-quality recording
- **Content:** Continuous speech, not silence or noise
- **Format:** Same as input audio formats

### Performance Optimization

#### Model Selection
- **Speed vs Quality:** Smaller models (tiny, base) are faster but less accurate
- **Memory Usage:** Larger models require more RAM
- **Hardware Acceleration:** Some models support GPU acceleration

#### Processing Options
- **Batch Processing:** Process multiple files sequentially
- **Quality Settings:** Adjust for speed vs quality trade-offs
- **Memory Management:** Configure cache and temporary file handling

## Troubleshooting

### Common Issues

#### Model Loading Errors
**Symptoms:** Models not appearing in dropdown, "Model not found" errors
**Solutions:**
1. Verify model files are in correct directories
2. Check file permissions
3. Ensure model file integrity
4. Restart application after adding new models

#### Audio Processing Failures
**Symptoms:** Processing stops unexpectedly, corrupted output
**Solutions:**
1. Verify audio file format compatibility
2. Check for corrupted audio files
3. Ensure sufficient disk space
4. Try different audio format

#### Memory Issues
**Symptoms:** Application crashes, "Out of memory" errors
**Solutions:**
1. Use smaller models
2. Close other applications
3. Increase virtual memory
4. Process shorter audio segments

#### Voice Cloning Quality
**Symptoms:** Poor voice similarity, robotic output
**Solutions:**
1. Use high-quality reference audio
2. Ensure reference audio is 6-10 seconds
3. Try different XTTS model versions
4. Check reference audio volume levels

### Debugging Steps

1. **Check Dependencies:**
   ```bash
   pip list | grep -E "(torch|transformers|tts|pyqt)"
   ```

2. **Verify Model Files:**
   ```bash
   ls -la Models/whisper/
   ls -la Models/nllb/
   ls -la Models/xtts/
   ```

3. **Test Individual Modules:**
   Use `minimal_test.py` to verify UI functionality independently

4. **Review Logs:**
   Check the application log area for specific error messages

### Performance Monitoring

Monitor the following during processing:
- Memory usage
- CPU utilization
- Disk I/O
- Processing time per stage

### Support Resources

- Check the README for installation instructions
- Verify all prerequisites are met
- Ensure models are correctly downloaded and placed
- Consult the troubleshooting section for common issues