#!/usr/bin/env python3
"""
Smoke test for OpenMOSS migration.
Tests basic imports and configuration without requiring models.
"""

import sys
import os

# Add project root to path (this file lives one level under the project root)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print("=" * 70)
print("MOSS Migration Smoke Test")
print("=" * 70)

# Test 1: Import setup checker
print("\n[1/5] Testing model_setup_checker import...")
try:
    from src.utils.model_setup_checker import (
        detect_system, select_tier, run_setup_check,
        get_stt_model_path, get_tts_model_path
    )
    print("  ✓ model_setup_checker imported successfully")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

# Test 2: Check system detection
print("\n[2/5] Testing system detection...")
try:
    profile = detect_system()
    print(f"  ✓ System detected:")
    print(f"    - CUDA available: {profile.has_cuda}")
    print(f"    - VRAM: {profile.vram_gb:.1f} GB")
    print(f"    - RAM: {profile.ram_gb:.1f} GB")
    print(f"    - CPU cores: {profile.cpu_cores}")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

# Test 3: Model tier selection
print("\n[3/5] Testing model tier selection...")
try:
    tier = select_tier(profile)
    print(f"  ✓ Selected tier: {tier.label}")
    print(f"    - STT model: {tier.stt_hf_id}")
    print(f"    - TTS model: {tier.tts_hf_id}")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

# Test 4: Import services
print("\n[4/5] Testing service imports...")
try:
    from src.core.services.transcription_service import Transcriber
    from src.core.services.voice_synthesis_service import VoiceSynthesizer
    from src.core.services.translation_service import Translator
    print("  ✓ All services imported successfully")
    print("    - Transcriber (MOSS-Audio)")
    print("    - VoiceSynthesizer (MOSS-TTS)")
    print("    - Translator (NLLB)")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

# Test 5: Import application components
print("\n[5/5] Testing application imports...")
try:
    from src.core.application.audio_orchestrator import AudioDubbingOrchestrator
    from src.core.services.model_manager import ModelManager
    print("  ✓ Application components imported successfully")
    print("    - AudioDubbingOrchestrator")
    print("    - ModelManager (with MOSS model scanning)")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("All smoke tests passed! ✓")
print("=" * 70)
print("\nNext steps:")
print("1. Install models using the setup checker (it will show instructions)")
print("2. Run 'python main.py' to start the application")
print("3. Models will be downloaded automatically on first run if missing")
