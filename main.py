#!/usr/bin/env python3
"""
Offline Audio Dubbing - Main Entry Point

This is the main entry point for the Offline Audio Dubbing application.
"""
import sys
import os

# Add the project root to the path to allow imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from PyQt5.QtWidgets import QApplication
import logging

# Hook setup checker before importing GUI
from src.utils.model_setup_checker import run_setup_check


def main():
    """Main entry point for the application."""
    models_ready, tier = run_setup_check(show_gui_dialog=True)
    if not models_ready:
        sys.exit(0)

    from src.api.interfaces.gui_interface import MainWindow

    app = QApplication(sys.argv)

    # Set application font
    from PyQt5.QtGui import QFont
    font = QFont("Times New Roman", 10)
    app.setFont(font)

    try:
        window = MainWindow()
        window.show()
        logging.info("Application started successfully")
        sys.exit(app.exec_())
    except Exception as e:
        logging.error(f"Application error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
