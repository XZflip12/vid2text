# URL Audio Transcription (yt-dlp + faster-whisper)

A small Python script to:

1. Download audio from a supported URL via `yt-dlp` (or use a local audio file).
2. Transcribe speech to text with `faster-whisper`.
3. Save the transcript with timestamps to `transcript.txt`.

The project supports GPU acceleration on Windows (CUDA), with automatic fallback to CPU if GPU runtime loading fails.

## Project Structure

- `main.py` - main script.
- `app/` - program modules.
- `models/` - local Whisper model cache (ignored by git).
- `cuda_dll/` - optional local CUDA DLL directory (ignored by git).
- `tempaudio.mp3` - temporary downloaded audio (kept or removed by your choice).
- `transcriptold.txt` - transcription output.

## Requirements

- Windows (current script paths are Windows-oriented).
- Python 3.
- FFmpeg installed.
- Internet access for downloading video/audio and model files (first run).

Python packages:

- `faster-whisper`
- `yt-dlp`

## 1) Setup (CPU-first, minimal)

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install required packages:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Set FFmpeg path in `main.py` (constant `DEFAULT_FFMPEG_PATH`).

## 2) Optional CUDA Setup (skip this for CPU mode)

If you want GPU mode, prepare CUDA runtime DLLs.
If they are not installed from requirements.txt:

### Option A: Use pip NVIDIA runtime packages inside `.venv`

```powershell
python -m pip install nvidia-cublas-cu12
python -m pip install nvidia-cudnn-cu12
python -m pip install nvidia-cuda-runtime-cu12
python -m pip install nvidia-cuda-nvrtc-cu12
```

### Option B: Put DLLs manually into project folder

Copy required DLL files into `cuda_dll/` (or `cuda_dll/bin`):

- `cublas64_12.dll`
- `cublasLt64_12.dll`
- `cudnn64_9.dll`
- `cudart64_12.dll`

The script automatically registers these directories at startup and prints diagnostics.

## 3) Configure Input

Use either interactive menu mode or CLI arguments (URL/local file).
Set `DEFAULT_FFMPEG_PATH` in `trans.py` if needed, or pass `--ffmpeg-path` at runtime.

## 4) Run

```powershell
python .\trans.py
```

At runtime, the script will:

- Try GPU with several compute types.
- Fall back to CPU automatically on CUDA runtime errors.
- Print transcription progress in the console.
- Ask whether to delete `tempaudio.mp3` after completion.
- Reuse existing `tempaudio.mp3` on next run (if it was kept).

## Output

- Main output file: `transcriptold.txt`
- Format per line: `[start -> end] text`

## Troubleshooting

### `cublas64_12.dll is not found or cannot be loaded`

1. Make sure `cu12` runtime files are available (especially `cudart64_12.dll`).
2. Check startup logs from `register_nvidia_dll_dirs()` and `diagnose_cuda_dlls()`.
3. Verify DLLs exist in `.venv\Lib\site-packages\nvidia\...\bin` or in `cuda_dll/`.

### GPU is still not used

- Confirm your NVIDIA driver is installed and compatible.
- Keep CUDA setup versions consistent (the script currently expects `cu12`-style DLL names).
- If needed, run CPU mode (CUDA setup can be skipped).

## Notes

- `models/` and `cuda_dll/` are intentionally excluded from git via `.gitignore`.
- Current script is single-file and intended for local usage.

