# Offline Audio Dubbing - Agent Configuration & Context Guide

**Last Updated**: May 2026  
**Status**: Production-Ready (v2.1 - Enhanced Architecture)

## Project Overview

**Offline Audio Dubbing** is a professional-grade desktop application for translating and dubbing audio files completely offline. It uses state-of-the-art deep learning models (Whisper, NLLB-200, XTTS-v2) to transcribe, translate, and synthesize audio with voice cloning.

The architecture follows a clean separation of concerns, with a central orchestrator managing the workflow between specialized services.

## Core Components

### Core Components

| Component | File | Purpose | Status |
|-----------|------|---------|--------|
| **ModelManager** | `src/core/services/model_manager.py` | Centralized model scanning and validation | ✅ New |
| **Orchestrator** | `src/core/application/audio_orchestrator.py` | Workflow coordination and business logic | ✅ Centralized |
| **Transcriber** | `src/core/services/transcription_service.py` | Speech-to-text using Whisper.cpp | ✅ Optimized |
| **Translator** | `src/core/services/translation_service.py` | Text translation using NLLB-200 | ✅ Optimized |
| **VoiceCloner** | `src/core/services/voice_synthesis_service.py` | Voice synthesis with cloning using XTTS-v2 | ✅ Optimized |
| **GUI Interface** | `src/api/interfaces/gui_interface.py` | PyQt5 main window, strictly UI-focused | ✅ Refactored |
| **Dialogs** | `src/api/interfaces/dialogs.py` | UI Dialog components | ✅ Refactored |
| **Helpers** | `src/utils/common/helpers.py` | Utilities, language support, validation | ✅ Cleaned |
| **Configuration** | `src/utils/common/app_config.py` | Settings, constants, paths | ✅ Enhanced |
| **Data Models** | `src/core/data_models/audio_models.py` | Type-safe data structures | ✅ Finalized |

## Architecture & Data Flow

### 1. Model Management
The `ModelManager` class provides a single source of truth for model availability. It scans the `Models/` directory and validates model files for Whisper, NLLB, and XTTS. This centralized logic is used by both the GUI (for selection) and the processing services.

### 2. Application Orchestration
The `AudioDubbingOrchestrator` manages the end-to-end processing pipeline. It is decoupled from the UI, accepting a configuration object and reporting progress/status via callbacks. It supports three input entry points:
- **Audio File**: Full pipeline (Transcribe → Translate → Dub)
- **Transcription Text**: Partial pipeline (Translate → Dub)
- **Translation Text**: Final step (Dub only)

### 3. Processing Pipeline (Audio Input)
```
Audio Input
    ↓ (Whisper)
Text Transcription (with timestamps)
    ↓ (Cleaning)
Plain Text
    ↓ (NLLB)
Translated Text
    ↓ (XTTS-v2)
Dubbed Audio Output (cloned voice)
```

## Language System

### Supported Languages
- Supports all 205+ NLLB-200 languages.
- Mapping between language names and codes is managed in `src/utils/common/helpers.py`.
- Deterministic language-to-number mapping ensures stable UI state.

## Error Handling & Robustness

- **Type Safety**: Full type hints used throughout the codebase.
- **Defensive Programming**: Extensive input validation for audio files, model paths, and language codes.
- **Custom Exceptions**: Specific exception types for different service failures.
- **Logging**: Comprehensive logging at DEBUG and INFO levels, with both console and file output.

## Directory Structure

```
offline-audio-dubbing/
├── main.py                              # Entry point
├── src/
│   ├── api/                             # UI Layer
│   │   └── interfaces/                  # GUI and Dialogs
│   ├── core/                            # Business Layer
│   │   ├── application/                 # Orchestration
│   │   ├── data_models/                 # Shared data structures
│   │   └── services/                    # Core AI services
│   └── utils/                           # Infrastructure Layer
│       └── common/                      # Helpers and config
├── Models/                              # Model files (git-ignored)
├── Inputs/                              # Input files (git-ignored)
└── Outputs/                             # Output files (git-ignored)
```

## Maintenance Guidelines

- **Adding a Service**: Implement the service in `src/core/services/` and integrate it into `AudioDubbingOrchestrator`.
- **UI Changes**: Modify `gui_interface.py` for general layout or `dialogs.py` for specialized windows.
- **Model Validation**: Update `ModelManager` if model file requirements change.
- **Language Support**: Update `get_nllb_languages` in `helpers.py`.
