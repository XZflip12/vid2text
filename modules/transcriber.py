import ctypes
import os
import sys
from pathlib import Path
from typing import List, Tuple
from faster_whisper import WhisperModel

from modules.settings import LOCAL_CUDA_DLL_DIR, LOCAL_MODEL_DIR, AppConfig
from modules.localizer import Localizer

GPU_COMPUTE_TYPES = ("float16", "int8_float16", "int8")
REQUIRED_CUDA_DLLS = ("cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll", "cudart64_12.dll")
DLL_DIR_HANDLES = []
localizer = Localizer()


def register_nvidia_dll_dirs() -> List[Path]:
    if sys.platform != "win32":
        return []

    LOCAL_CUDA_DLL_DIR.mkdir(parents=True, exist_ok=True)
    candidate_dirs = [LOCAL_CUDA_DLL_DIR]
    if (LOCAL_CUDA_DLL_DIR / "bin").exists():
        candidate_dirs.append(LOCAL_CUDA_DLL_DIR / "bin")

    nvidia_root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if nvidia_root.exists():
        for root, dirs, _ in os.walk(nvidia_root):
            if "bin" in dirs:
                candidate_dirs.append(Path(root) / "bin")

    env_cuda_paths = [Path(value) / "bin" for key, value in os.environ.items() if key.startswith("CUDA_PATH") and value]
    candidate_dirs.extend(env_cuda_paths)

    seen = set()
    registered_dirs = []
    for dll_dir in candidate_dirs:
        normalized = str(dll_dir.resolve()) if dll_dir.exists() else str(dll_dir)
        if normalized in seen or not dll_dir.exists():
            continue
        seen.add(normalized)
        try:
            DLL_DIR_HANDLES.append(os.add_dll_directory(str(dll_dir)))
            os.environ["PATH"] = f"{dll_dir}{os.pathsep}" + os.environ.get("PATH", "")
            registered_dirs.append(dll_dir)

            print(localizer.get_string('dll_path_added', dll_dir))
        except OSError:
            pass
    return registered_dirs

def diagnose_cuda_dlls() -> None:
    if sys.platform != "win32":
        return
    dirs = register_nvidia_dll_dirs()
    print(localizer.get_string('choice_diagnostics'))
    missing = []
    for dll_name in REQUIRED_CUDA_DLLS:
        found = False
        for search_dir in dirs:
            dll_path = search_dir / dll_name
            if dll_path.exists():
                found = True
                try:
                    ctypes.WinDLL(str(dll_path))
                    print(f"[OK] {dll_name}: {dll_path}")
                except OSError as exc:
                    print(f"[FAIL LOAD] {dll_name}: {dll_path} ({exc})")
                break
        if not found:
            missing.append(dll_name)
            print(f"[MISSING] {dll_name}")

    if missing:
        print(localizer.get_string('missing_cuda_dlls'))

def create_whisper_model(model_size: str, prefer_gpu: bool = True) -> Tuple[WhisperModel, str]:
    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if prefer_gpu:
        for compute_type in GPU_COMPUTE_TYPES:
            try:
                model = WhisperModel(model_size, device="cuda", compute_type=compute_type, download_root=str(LOCAL_MODEL_DIR))
                print(localizer.get_string('gpu_launch', compute_type))
                return model, "cuda"
            except Exception as exc:
                print(localizer.get_string('error_cuda_init', compute_type, exc))

    print(localizer.get_string('cpu_fallback_notice'))
    return WhisperModel(model_size, device="cpu", compute_type="int8", download_root=str(LOCAL_MODEL_DIR)), "cpu"

def is_cuda_runtime_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in ("cublas", "cudnn", "cuda", "cannot be loaded", "is not found"))

def collect_transcript_lines(model: WhisperModel, audio_filename: str, config: AppConfig) -> str:
    last_percent = -1
    segments, info = model.transcribe(audio_filename, beam_size=config.beam_size, language=config.language, vad_filter=True)
    total_duration = getattr(info, "duration", None) or getattr(info, "duration_after_vad", None)

    with open(config.output_file, "w", encoding="utf-8") as f:
        for idx, segment in enumerate(segments, start=1):
            line = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n"
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

            if total_duration and total_duration > 0:
                percent = min(100, int((segment.end / total_duration) * 100))
                if percent != last_percent:
                    print(localizer.get_string('progress_bar', percent, segment.end, total_duration), end='')
                    last_percent = percent
            else:
                print(localizer.get_string('segment_processed', idx), end='')
    print(localizer.get_string('transcription_finished'))

def transcribe_with_fallback(audio_filename: str, config: AppConfig) -> str:
    model, device = create_whisper_model(config.model_size, prefer_gpu=config.device_mode == "auto")
    try:
        collect_transcript_lines(model, audio_filename, config)
        return device
    except RuntimeError as exc:
        if device == "cuda" and is_cuda_runtime_error(exc):
            print(localizer.get_string('error_cuda_runtime', exc))
            cpu_model, _ = create_whisper_model(config.model_size, prefer_gpu=False)
            collect_transcript_lines(cpu_model, audio_filename, config)
            return "cpu"
        raise
