Here is the updated `README.md` for your new modular architecture.

---

# URL Audio Transcription (Modular Architecture)

A robust Python tool to transcribe audio to text using `faster-whisper`. It features a modular architecture, persistent JSON settings, real-time transcription saving to prevent data loss, and automatic GPU/CPU fallback.

## Project Structure

The project is organized into logical modules for better maintainability:

```text
project_root/
├── app.py              # Main entry point and CLI menu
├── settings.json       # Persistent user configuration
├── modules/
│   ├── __init__.py
│   ├── settings.py     # Settings management (JSON handling)
│   ├── downloader.py   # yt-dlp integration and file management
│   └── transcriber.py  # Faster-whisper logic and CUDA diagnostics
├── models/             # Local Whisper model cache
└── cuda_dll/           # Optional local CUDA DLL directory
```

## Requirements

* **Windows** (Optimized for Windows path handling).
* **Python 3.10+**.
* **FFmpeg** installed (provide the path in settings).
* **Internet access** (for model download and video fetching).

## Installation

1. **Clone/Download** the repository.
2. **Create and activate a virtual environment:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. **Install dependencies:**

```powershell
python -m pip install --upgrade pip
python -m pip install faster-whisper yt-dlp
```

4. **Optional CUDA (GPU) Setup:**
To enable GPU acceleration, install the necessary NVIDIA runtime packages within your virtual environment:

```powershell
python -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12
```

## Usage

Start the application by running:

```powershell
python app.py
```

### Features

* **Modular Architecture:** Easy to extend or modify.
* **JSON Settings:** Your preferences (FFmpeg path, model size, language, etc.) are saved automatically in `settings.json`.
* **Real-time Saving:** The transcript is written to disk segment-by-segment as it is processed, protecting your work against crashes.
* **Interactive CLI:** An intuitive menu to switch between URL or local file processing and diagnostic tools.
* **CUDA Resilience:** Automatic registration of DLL directories and fallback to CPU if GPU initialization fails.

## Diagnostics & Troubleshooting

If you encounter issues with GPU acceleration:

1. Use the **"Диагностика CUDA DLL"** option (Option 4) in the main menu. This will verify if the required DLLs are found by the system.
2. Ensure you have the correct NVIDIA drivers installed.
3. If `cublas64_12.dll` or similar files are reported as missing, you can manually place them in the `cuda_dll/` folder, and the script will automatically include them in the search path.

## Output

* **Transcripts:** Saved as text files (default `transcript_old.txt`).
* **Format:** `[start_time -> end_time] transcribed text`

## Notes

* `models/`, `cuda_dll/`, `settings.json`, and `__pycache__/` are automatically handled and usually ignored by Git.
* The `app.py` script includes a forced exit mechanism (`os._exit(0)`) to cleanly bypass common CTranslate2 memory cleanup issues on Windows.