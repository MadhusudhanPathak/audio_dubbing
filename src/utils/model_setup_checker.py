"""
model_setup_checker.py
Detects system capabilities, determines the correct OpenMOSS model tier,
creates required model directories, and prints actionable download instructions
if any models are missing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import NamedTuple, Optional

import psutil
import torch


# ── Canonical model directory layout (relative to project root) ───────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR   = PROJECT_ROOT / "Models"

REQUIRED_DIRS: dict[str, Path] = {
    # STT
    "moss_audio":               MODELS_DIR / "moss-audio",
    # TTS
    "moss_tts":                 MODELS_DIR / "moss-tts",
    "moss_audio_tokenizer":     MODELS_DIR / "moss-audio-tokenizer",
    # Nano (fallback)
    "moss_tts_nano":            MODELS_DIR / "moss-tts-nano",
    "moss_audio_tokenizer_nano":MODELS_DIR / "moss-audio-tokenizer-nano",
    # GGUF (mid-range GPU fallback)
    "moss_tts_gguf":            MODELS_DIR / "moss-tts-gguf",
    "moss_audio_tokenizer_onnx":MODELS_DIR / "moss-audio-tokenizer-onnx",
}


class SystemProfile(NamedTuple):
    has_cuda:      bool
    vram_gb:       float
    ram_gb:        float
    cpu_cores:     int
    gpu_name:      str


class ModelTier(NamedTuple):
    label:         str          # human-readable tier name
    stt_dir:       str          # key into REQUIRED_DIRS
    stt_hf_id:     str          # HuggingFace model ID for STT
    tts_dir:       str          # key into REQUIRED_DIRS
    tts_hf_id:     str          # HuggingFace model ID for TTS
    extra_dirs:    list[str]    # additional required dirs (tokenizers, etc.)
    extra_hf_ids:  list[str]    # corresponding HF IDs


_cached_profile_and_tier: Optional[tuple["SystemProfile", "ModelTier"]] = None


def detect_system() -> SystemProfile:
    has_cuda  = torch.cuda.is_available()
    vram_gb   = 0.0
    gpu_name  = "None"
    if has_cuda:
        props    = torch.cuda.get_device_properties(0)
        vram_gb  = props.total_memory / (1024 ** 3)
        gpu_name = props.name
    ram_gb    = psutil.virtual_memory().total / (1024 ** 3)
    cpu_cores = os.cpu_count() or 1
    return SystemProfile(has_cuda, vram_gb, ram_gb, cpu_cores, gpu_name)


def select_tier(profile: SystemProfile) -> ModelTier:
    """
    Choose the best model combination for the detected hardware.

    Tier logic (models are loaded sequentially, not simultaneously):
      CPU / <6 GB VRAM  → Nano TTS  + Audio-4B-Instruct (CPU/quantized)
      6–10 GB VRAM      → GGUF TTS  + Audio-4B-Instruct
      10–20 GB VRAM     → TTS-Local + Audio-4B-Instruct
      20–28 GB VRAM     → TTS-8B    + Audio-4B-Instruct
      28+ GB VRAM       → TTS-8B    + Audio-8B-Instruct
    """
    v = profile.vram_gb

    if not profile.has_cuda or v < 6:
        return ModelTier(
            label        = "CPU / Low VRAM (≤6 GB) — Nano TTS + Audio-4B-Instruct",
            stt_dir      = "moss_audio",
            stt_hf_id    = "OpenMOSS-Team/MOSS-Audio-4B-Instruct",
            tts_dir      = "moss_tts_nano",
            tts_hf_id    = "OpenMOSS-Team/MOSS-TTS-Nano",
            extra_dirs   = ["moss_audio_tokenizer_nano"],
            extra_hf_ids = ["OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"],
        )
    elif v < 10:
        return ModelTier(
            label        = "Mid GPU (6–10 GB VRAM) — GGUF TTS + Audio-4B-Instruct",
            stt_dir      = "moss_audio",
            stt_hf_id    = "OpenMOSS-Team/MOSS-Audio-4B-Instruct",
            tts_dir      = "moss_tts_gguf",
            tts_hf_id    = "OpenMOSS-Team/MOSS-TTS-GGUF",
            extra_dirs   = ["moss_audio_tokenizer_onnx"],
            extra_hf_ids = ["OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX"],
        )
    elif v < 20:
        return ModelTier(
            label        = "Mid-High GPU (10–20 GB VRAM) — TTS-Local + Audio-4B-Instruct",
            stt_dir      = "moss_audio",
            stt_hf_id    = "OpenMOSS-Team/MOSS-Audio-4B-Instruct",
            tts_dir      = "moss_tts",
            tts_hf_id    = "OpenMOSS-Team/MOSS-TTS-Local-Transformer",
            extra_dirs   = ["moss_audio_tokenizer"],
            extra_hf_ids = ["OpenMOSS-Team/MOSS-Audio-Tokenizer"],
        )
    elif v < 28:
        return ModelTier(
            label        = "High GPU (20–28 GB VRAM) — TTS-8B + Audio-4B-Instruct",
            stt_dir      = "moss_audio",
            stt_hf_id    = "OpenMOSS-Team/MOSS-Audio-4B-Instruct",
            tts_dir      = "moss_tts",
            tts_hf_id    = "OpenMOSS-Team/MOSS-TTS",
            extra_dirs   = ["moss_audio_tokenizer"],
            extra_hf_ids = ["OpenMOSS-Team/MOSS-Audio-Tokenizer"],
        )
    else:
        return ModelTier(
            label        = "Flagship GPU (28+ GB VRAM) — TTS-8B + Audio-8B-Instruct",
            stt_dir      = "moss_audio",
            stt_hf_id    = "OpenMOSS-Team/MOSS-Audio-8B-Instruct",
            tts_dir      = "moss_tts",
            tts_hf_id    = "OpenMOSS-Team/MOSS-TTS",
            extra_dirs   = ["moss_audio_tokenizer"],
            extra_hf_ids = ["OpenMOSS-Team/MOSS-Audio-Tokenizer"],
        )


def _model_present(dir_key: str) -> bool:
    """A model directory is considered populated if config.json exists inside it."""
    return (REQUIRED_DIRS[dir_key] / "config.json").exists()


def create_missing_directories(tier: ModelTier) -> list[str]:
    """Create all directories this tier needs. Return list of dirs that were created."""
    needed_keys = [tier.stt_dir, tier.tts_dir] + tier.extra_dirs
    created = []
    for key in needed_keys:
        path = REQUIRED_DIRS[key]
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created.append(str(path))
    return created


def build_download_instructions(tier: ModelTier, profile: SystemProfile) -> str:
    """Return a formatted, copy-pasteable instruction block."""
    all_dirs    = [tier.stt_dir, tier.tts_dir] + tier.extra_dirs
    all_hf_ids  = [tier.stt_hf_id, tier.tts_hf_id] + tier.extra_hf_ids
    missing     = [(d, h) for d, h in zip(all_dirs, all_hf_ids) if not _model_present(d)]

    if not missing:
        return ""

    lines = [
        "",
        "=" * 70,
        "  MOSS-Audio Dubbing — Model Setup Required",
        "=" * 70,
        f"  Detected hardware:  {profile.gpu_name or 'CPU only'}",
        f"  VRAM:               {profile.vram_gb:.1f} GB" if profile.has_cuda else
        f"  VRAM:               N/A (CPU mode)",
        f"  RAM:                {profile.ram_gb:.1f} GB",
        f"  CPU cores:          {profile.cpu_cores}",
        f"  Selected tier:      {tier.label}",
        "",
        "  The following model directories are missing or empty.",
        "  Run these commands from your terminal, then restart the app:",
        "",
    ]

    for dir_key, hf_id in missing:
        abs_path = REQUIRED_DIRS[dir_key].resolve()
        lines += [
            f"  ── {hf_id} ──",
            f"     Download page : https://huggingface.co/{hf_id}",
            f"     Put files in  : {abs_path}",
            f"     Quick command :",
            f"       huggingface-cli download {hf_id} \\",
            f"           --local-dir \"{abs_path}\"",
            "",
        ]

    lines += [
        "  After downloading, verify each directory contains a config.json file.",
        "=" * 70,
        "",
    ]
    return "\n".join(lines)


def run_setup_check(show_gui_dialog: bool = False) -> tuple[bool, ModelTier]:
    """
    Run the full startup check.

    Returns:
        (all_models_present: bool, selected_tier: ModelTier)

    If `show_gui_dialog=True` and PyQt5 is available, shows a GUI dialog
    instead of printing to stdout — useful when called from the GUI entry point.
    """
    global _cached_profile_and_tier
    if _cached_profile_and_tier is None:
        profile = detect_system()
        tier    = select_tier(profile)
        _cached_profile_and_tier = (profile, tier)
    else:
        profile, tier = _cached_profile_and_tier

    created = create_missing_directories(tier)
    if created:
        print(f"[setup] Created model directories: {', '.join(created)}")

    instructions = build_download_instructions(tier, profile)

    if instructions:
        if show_gui_dialog:
            _show_qt_dialog(instructions, tier, profile)
        else:
            print(instructions, file=sys.stderr)
        return False, tier

    return True, tier


def _show_qt_dialog(message: str, tier: ModelTier, profile: SystemProfile) -> None:
    """Show a user-friendly PyQt5 dialog with download instructions."""
    try:
        from PyQt5.QtWidgets import (
            QApplication, QDialog, QVBoxLayout, QHBoxLayout,
            QLabel, QPushButton, QTextBrowser
        )
        from PyQt5.QtGui import QFont

        app = QApplication.instance() or QApplication(sys.argv)

        dialog = QDialog()
        dialog.setWindowTitle("🎬 OpenMOSS Model Setup Required")
        dialog.setGeometry(100, 100, 900, 700)
        dialog.setModal(True)

        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Header section with hardware info
        header_font = QFont()
        header_font.setPointSize(12)
        header_font.setBold(True)

        header = QLabel("System Configuration & Model Requirements")
        header.setFont(header_font)
        layout.addWidget(header)

        # Hardware details
        hw_text = (
            f"<b>Detected Hardware:</b> {profile.gpu_name or 'CPU only'}<br>"
            f"<b>GPU VRAM:</b> {profile.vram_gb:.1f} GB<br>"
            f"<b>System RAM:</b> {profile.ram_gb:.1f} GB<br>"
            f"<b>CPU Cores:</b> {profile.cpu_cores}<br><br>"
            f"<b style='color: #0066cc;'>Recommended Tier:</b> {tier.label}"
        )
        hw_label = QLabel(hw_text)
        hw_label.setStyleSheet(
            "background-color: #f0f8ff; border-left: 4px solid #0066cc; padding: 10px;"
        )
        layout.addWidget(hw_label)

        # Separator
        sep = QLabel("")
        sep.setStyleSheet("border-bottom: 1px solid #cccccc;")
        layout.addWidget(sep)

        # Instructions header
        inst_header = QLabel("Model Download Instructions")
        inst_header.setFont(header_font)
        layout.addWidget(inst_header)

        # Text browser for instructions (with clickable links)
        text_browser = QTextBrowser()
        text_browser.setHtml(_format_html_instructions(message))
        text_browser.setOpenExternalLinks(True)
        text_browser.setFont(QFont("Courier", 9))
        text_browser.setStyleSheet(
            "background-color: #f5f5f5; border: 1px solid #ddd; padding: 10px;"
        )
        layout.addWidget(text_browser)

        # Bottom section with buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        copy_btn = QPushButton("📋 Copy All Commands")
        copy_btn.setMinimumHeight(40)
        copy_btn.setMinimumWidth(200)
        copy_btn.clicked.connect(lambda: _copy_to_clipboard(message))

        ok_btn = QPushButton("✓ OK, I'll Download the Models")
        ok_btn.setMinimumHeight(40)
        ok_btn.setMinimumWidth(200)
        ok_btn.setStyleSheet(
            "QPushButton { background-color: #0066cc; color: white; "
            "font-weight: bold; border-radius: 4px; }"
        )
        ok_btn.clicked.connect(dialog.accept)

        button_layout.addWidget(copy_btn)
        button_layout.addStretch()
        button_layout.addWidget(ok_btn)

        layout.addLayout(button_layout)

        dialog.setLayout(layout)
        dialog.exec_()

    except Exception as e:
        print(message, file=sys.stderr)
        print(f"Error showing dialog: {e}", file=sys.stderr)


def _format_html_instructions(message: str) -> str:
    """Convert plain text instructions to formatted HTML."""
    html = "<div style='font-family: Courier, monospace; line-height: 1.6;'>"

    for line in message.split('\n'):
        line = line.rstrip()

        if not line:
            html += "<br>"
        elif line.startswith('  ──'):
            # Model header
            model_name = line.replace('  ── ', '').replace(' ──', '')
            html += f"<div style='margin-top: 15px; margin-bottom: 5px;'>"
            html += f"<span style='background-color: #e6f2ff; padding: 5px 10px; "
            html += f"border-radius: 3px; font-weight: bold; color: #0066cc;'>{model_name}</span>"
            html += f"</div>"
        elif 'huggingface.co' in line:
            # Extract URL and make it clickable
            parts = line.split('https://')
            if len(parts) > 1:
                url = 'https://' + parts[1].split('<')[0]
                label = url.replace('https://huggingface.co/', '')
                html += f"<div style='margin: 8px 0;'>"
                html += f"<span style='color: #666;'>📥 Download:</span> "
                html += f"<a href='{url}' style='color: #0066cc; text-decoration: none;'>{label}</a>"
                html += f"</div>"
            else:
                html += f"<div style='margin: 8px 0; color: #666;'>{line}</div>"
        elif 'Put files in' in line:
            # Directory path
            parts = line.split(':')
            if len(parts) > 1:
                path = parts[1].strip()
                html += f"<div style='margin: 8px 0;'>"
                html += f"<span style='color: #666;'>📁 Destination:</span> "
                html += f"<code style='background-color: #fff9e6; padding: 3px 6px; "
                html += f"border-radius: 3px;'>{path}</code>"
                html += f"</div>"
        elif 'Quick command' in line:
            html += f"<div style='margin: 10px 0; color: #666;'><strong>⚡ Quick Command:</strong></div>"
        elif line.strip().startswith('huggingface-cli'):
            # Command to run
            html += f"<div style='background-color: #1e1e1e; color: #00ff00; padding: 10px; "
            html += f"border-radius: 3px; margin: 8px 0; overflow-x: auto;'>"
            html += f"<code>{line.strip()}</code>"
            html += f"</div>"
        elif line.startswith('='):
            # Separator
            html += f"<hr style='border: 1px solid #ddd; margin: 15px 0;'>"
        elif line.strip().startswith('--'):
            # Flag continuation
            html += f"<div style='margin-left: 20px; margin: 2px 0;'><code>{line.strip()}</code></div>"
        elif 'config.json' in line:
            # Verification info
            html += f"<div style='margin: 8px 0; color: #666; font-size: 0.9em;'>"
            html += f"<span style='color: #009900;'>✓</span> {line.strip()}"
            html += f"</div>"
        else:
            html += f"<div style='margin: 2px 0; color: #666;'>{line}</div>"

    html += "</div>"
    return html


def _copy_to_clipboard(text: str) -> None:
    """Copy text to clipboard and show confirmation."""
    try:
        from PyQt5.QtWidgets import QApplication
        clipboard = QApplication.clipboard()
        clipboard.setText(text)

        # Show confirmation
        from PyQt5.QtWidgets import QMessageBox
        msg = QMessageBox()
        msg.setWindowTitle("✓ Copied")
        msg.setText("Download instructions copied to clipboard!")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
    except Exception as e:
        print(f"Error copying to clipboard: {e}", file=sys.stderr)


def get_stt_model_path() -> str:
    _, tier = run_setup_check()
    return str(REQUIRED_DIRS[tier.stt_dir])


def get_tts_model_path() -> str:
    _, tier = run_setup_check()
    return str(REQUIRED_DIRS[tier.tts_dir])
