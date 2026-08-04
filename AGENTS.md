# Offline Audio Dubbing - Agent Configuration & Context Guide

## Project Overview

**Offline Audio Dubbing** is a desktop application for translating and dubbing audio files completely offline. It uses local deep learning models (OpenMOSS MOSS-Audio, NLLB-200, OpenMOSS MOSS-TTS) to transcribe, translate, and synthesize audio with voice cloning.

The architecture follows a clean separation of concerns, with a central orchestrator managing the workflow between specialized services.

## Core Components

| Component | File | Purpose |
|-----------|------|---------|
| **ModelManager** | `src/core/services/model_manager.py` | Scans and validates locally available models |
| **Orchestrator** | `src/core/application/audio_orchestrator.py` | Workflow coordination and business logic |
| **Transcriber** | `src/core/services/transcription_service.py` | Speech-to-text using OpenMOSS MOSS-Audio |
| **Translator** | `src/core/services/translation_service.py` | Text translation using NLLB-200 |
| **VoiceSynthesizer** | `src/core/services/voice_synthesis_service.py` | Voice synthesis with cloning using OpenMOSS MOSS-TTS |
| **HFModelService** | `src/core/services/base_model_service.py` | Shared device/dtype and unload lifecycle for the STT/TTS services |
| **Model Setup Checker** | `src/utils/model_setup_checker.py` | Detects hardware, selects a model tier, and reports missing models at startup |
| **GUI Interface** | `src/api/interfaces/gui_interface.py` | PyQt5 main window, strictly UI-focused |
| **Dialogs** | `src/api/interfaces/dialogs.py` | UI dialog components |
| **Helpers** | `src/utils/common/helpers.py` | Utilities, language support, validation |
| **Configuration** | `src/utils/common/app_config.py` | Settings, constants, paths |
| **Data Models** | `src/core/data_models/audio_models.py` | Type-safe data structures |

## Architecture & Data Flow

### 1. Model Setup & Management
`model_setup_checker.py` runs at startup, detects the user's hardware (CUDA/VRAM/RAM), and selects the appropriate MOSS-Audio/MOSS-TTS model tier for that hardware. It reports any missing model directories before the app attempts to load anything. `ModelManager` provides on-demand scanning of what's actually present in `Models/` for the GUI's model-selection dropdowns.

### 2. Application Orchestration
The `AudioDubbingOrchestrator` manages the end-to-end processing pipeline. It is decoupled from the UI, accepting a configuration object and reporting progress/status via callbacks. It supports three input entry points:
- **Audio File**: Full pipeline (Transcribe -> Translate -> Dub)
- **Transcription Text**: Partial pipeline (Translate -> Dub)
- **Translation Text**: Final step (Dub only)

### 3. Processing Pipeline (Audio Input)
```
Audio Input
    -> (MOSS-Audio)
Text Transcription
    -> (Cleaning)
Plain Text
    -> (NLLB-200)
Translated Text
    -> (MOSS-TTS)
Dubbed Audio Output (cloned voice)
```

## Language System

- Supports the languages listed in `get_nllb_languages()` for dubbing (currently English, Italian, Hindi, German, Spanish, French).
- Mapping between language names and NLLB codes is managed in `src/utils/common/helpers.py`.

## Error Handling & Robustness

- **Type Safety**: Type hints used throughout the service and application layers.
- **Defensive Programming**: Input validation for audio files, model paths, and language codes; model load/unload wrapped in `try/finally` so GPU memory is freed even if a processing step fails.
- **Custom Exceptions**: `TranslationError` for translator-specific failures.
- **Logging**: Logging at DEBUG and INFO levels, with both console and file output.

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
│       ├── common/                      # Helpers and config
│       └── model_setup_checker.py       # Hardware detection & model tier selection
├── tests/                               # Manual verification scripts
├── Models/                              # Model files (git-ignored)
├── Inputs/                              # Input files (git-ignored)
└── Outputs/                             # Output files (git-ignored)
```

## Maintenance Guidelines

- **Adding a Service**: Implement the service in `src/core/services/` and integrate it into `AudioDubbingOrchestrator`.
- **UI Changes**: Modify `gui_interface.py` for general layout or `dialogs.py` for specialized windows.
- **Model Validation**: Update `ModelManager` if model file requirements change, or `model_setup_checker.py` if tier/hardware logic changes.
- **Language Support**: Update `get_nllb_languages` in `helpers.py`.
