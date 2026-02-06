# Troubleshooting Guide - Offline Audio Dubbing

## Common Issues and Solutions

### Installation Issues

#### Problem: Dependency Installation Fails
**Symptoms:** Errors during `pip install -r requirements.txt`

**Solutions:**
1. **Upgrade pip first:**
   ```bash
   python -m pip install --upgrade pip
   ```

2. **Install PyTorch separately (especially on Windows):**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Install packages individually if needed:**
   ```bash
   pip install PyQt5
   pip install transformers
   pip install coqui-tts
   # etc.
   ```

#### Problem: Microsoft Visual C++ Redistributables Missing (Windows)
**Symptoms:** DLL errors, PyTorch import errors

**Solution:**
1. Download and install Microsoft Visual C++ Redistributable for Visual Studio
2. Install both x64 and x86 versions
3. Link: https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads

### Model-Related Issues

#### Problem: Models Don't Appear in Dropdown Menus
**Symptoms:** "No models found" in dropdowns despite having models downloaded

**Solutions:**
1. **Verify directory structure:**
   ```
   Models/
   ├── whisper/          # .bin files go here
   ├── nllb/             # model folders go here  
   └── xtts/             # model folders go here
   ```

2. **Check file extensions:**
   - Whisper: `.bin` files (GGML format)
   - NLLB: Complete model folders
   - XTTS: Complete model folders

3. **Click "Refresh Models" button** after adding new models

4. **Verify file integrity:** Re-download if files are corrupted

#### Problem: Model Loading Fails
**Symptoms:** Error messages about model loading, application crashes when selecting models

**Solutions:**
1. **Check model compatibility:** Ensure models are compatible with the library versions
2. **Verify sufficient disk space:** Models can be several GB in size
3. **Check file permissions:** Ensure the application can read model files
4. **Try smaller models:** If using large models, try smaller ones first

### Audio Processing Issues

#### Problem: Audio Files Not Recognized
**Symptoms:** "Unsupported format" or "Invalid audio file" errors

**Solutions:**
1. **Verify supported formats:** WAV, MP3, FLAC, M4A, AAC, OGG, WMA
2. **Convert audio files** to a supported format using tools like ffmpeg
3. **Check file integrity:** Ensure audio files are not corrupted

#### Problem: Voice Cloning Produces Poor Quality
**Symptoms:** Robotic output, poor voice similarity, unnatural speech

**Solutions:**
1. **Improve reference audio:**
   - Use 6-10 seconds of clear speech
   - Ensure consistent volume
   - Minimize background noise
   - Use high-quality recording

2. **Try different XTTS models:** Different model versions may produce better results

3. **Adjust audio levels:** Ensure reference audio is neither too quiet nor too loud

#### Problem: Processing Takes Too Long
**Symptoms:** Very slow processing times

**Solutions:**
1. **Use smaller models:** Tiny or base Whisper models are faster
2. **Process shorter audio clips:** Split long files into smaller segments
3. **Close other applications:** Free up system resources
4. **Check hardware specs:** Ensure sufficient RAM and CPU power

### Memory and Performance Issues

#### Problem: Out of Memory Errors
**Symptoms:** "Out of memory" errors, application crashes during processing

**Solutions:**
1. **Use smaller models:** Switch to smaller Whisper/NLLB models
2. **Increase virtual memory:** Adjust system virtual memory settings
3. **Process smaller files:** Split large audio files
4. **Close other applications:** Free up RAM

#### Problem: Application Crashes During Processing
**Symptoms:** Unexpected application termination during processing

**Solutions:**
1. **Check system resources:** Monitor RAM and CPU usage
2. **Reduce processing load:** Use smaller models or shorter audio
3. **Update dependencies:** Ensure all packages are compatible
4. **Check logs:** Look at the log area for specific error messages

### Platform-Specific Issues

#### Windows Issues
1. **DLL errors:** Install Visual C++ Redistributables
2. **Path issues:** Ensure no spaces or special characters in installation path
3. **Antivirus interference:** Temporarily disable antivirus during installation

#### macOS Issues
1. **Permission errors:** Check file permissions for model directories
2. **SIP protection:** May need to grant additional permissions
3. **Xcode tools:** Install command line tools if needed

#### Linux Issues
1. **Missing libraries:** Install required system libraries
2. **Audio drivers:** Ensure proper audio subsystem (ALSA/PulseAudio)
3. **Permissions:** Check user permissions for audio devices

### Network and Download Issues

#### Problem: Cannot Download Models
**Symptoms:** Unable to access Hugging Face or other model repositories

**Solutions:**
1. **Check internet connection:** Ensure stable connection
2. **Firewall settings:** Check if corporate firewall is blocking access
3. **Alternative sources:** Look for mirrors or direct downloads
4. **VPN:** Try using a VPN if region-restricted

### Debugging Steps

#### Step 1: Verify Installation
```bash
# Check Python version
python --version

# Check key dependencies
python -c "import torch; print(torch.__version__)"
python -c "import PyQt5; print('PyQt5 OK')"
python -c "import TTS; print('TTS OK')"
```

#### Step 2: Test Individual Components
1. **Run minimal test:**
   ```bash
   python minimal_test.py
   ```
   If this works, UI is fine but there's an issue with ML components.

2. **Test model directories:**
   ```bash
   ls -la Models/whisper/
   ls -la Models/nllb/
   ls -la Models/xtts/
   ```

#### Step 3: Check Logs
- Look at the application's log area for specific error messages
- Check system logs if application crashes
- Enable verbose logging if available

#### Step 4: Isolate the Issue
1. **Try transcription only:** See if Whisper works independently
2. **Test with small audio file:** Rule out file size issues
3. **Use default settings:** Rule out configuration issues

### When Nothing Else Works

#### Reset Everything
1. **Delete virtual environment:**
   ```bash
   rm -rf venv  # or rmdir /s venv on Windows
   ```

2. **Clean install:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Re-download models**

4. **Try the minimal test again**

### Getting Help

If you're still experiencing issues:

1. **Check the logs** for specific error messages
2. **Search online** for the specific error message
3. **Create an issue** on the project repository with:
   - Your operating system
   - Python version
   - Specific error message
   - Steps you've tried
   - Your hardware specifications