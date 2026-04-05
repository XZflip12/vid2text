import os
import sys
import ctypes
from pathlib import Path
from typing import List, Tuple

from faster_whisper import WhisperModel
from yt_dlp import YoutubeDL

MODEL_SIZE = "medium"
TRANSCRIPT_FILE = "transcript.txt"
TEMP_AUDIO_TEMPLATE = "temp_audio.%(ext)s"
TEMP_AUDIO_FILE = "temp_audio.mp3"
DEFAULT_LANGUAGE = "ru"
DEFAULT_BEAM_SIZE = 5
DEFAULT_FFMPEG_PATH = r"C:\Users\oldfa\OneDrive\Документы\ffmpeg\bin\ffmpeg.exe"
GPU_COMPUTE_TYPES = ("float16", "int8_float16", "int8")
PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_MODEL_DIR = PROJECT_ROOT / "models"
LOCAL_CUDA_DLL_DIR = PROJECT_ROOT / "cuda_dll"
REQUIRED_CUDA_DLLS = (
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    "cudnn64_9.dll",
    "cudart64_12.dll",
)

# На Windows нужно удерживать handle add_dll_directory, иначе путь может быть снят.
DLL_DIR_HANDLES = []


def register_nvidia_dll_dirs() -> List[Path]:
    if sys.platform != "win32":
        return []

    LOCAL_CUDA_DLL_DIR.mkdir(parents=True, exist_ok=True)

    candidate_dirs = []

    # Локальная папка проекта для ручного размещения cuBLAS/cuDNN DLL.
    candidate_dirs.append(LOCAL_CUDA_DLL_DIR)
    local_bin_dir = LOCAL_CUDA_DLL_DIR / "bin"
    if local_bin_dir.exists():
        candidate_dirs.append(local_bin_dir)

    # Библиотеки nvidia из текущего venv (pip-пакеты ctranslate2/cudnn/cublas).
    nvidia_root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if nvidia_root.exists():
        for root, dirs, _ in os.walk(nvidia_root):
            if "bin" in dirs:
                candidate_dirs.append(Path(root) / "bin")

    # CUDA Toolkit из переменных окружения, если установлен системно.
    env_cuda_paths = [
        Path(value) / "bin"
        for key, value in os.environ.items()
        if key.startswith("CUDA_PATH") and value
    ]
    candidate_dirs.extend(env_cuda_paths)

    seen = set()
    registered_count = 0
    registered_dirs = []
    for dll_dir in candidate_dirs:
        normalized = str(dll_dir.resolve()) if dll_dir.exists() else str(dll_dir)
        if normalized in seen or not dll_dir.exists():
            continue
        seen.add(normalized)
        try:
            DLL_DIR_HANDLES.append(os.add_dll_directory(str(dll_dir)))
            os.environ["PATH"] = f"{dll_dir}{os.pathsep}" + os.environ.get("PATH", "")
            registered_count += 1
            registered_dirs.append(dll_dir)
            print(f"--- DLL-путь добавлен: {dll_dir} ---")
        except OSError as exc:
            print(f"--- Не удалось добавить DLL-путь: {dll_dir} ({exc}) ---")

    if registered_count == 0:
        print("--- DLL-пути не добавлены. Проверьте наличие папок с CUDA-библиотеками. ---")

    if not any((path / "cublas64_12.dll").exists() for path in candidate_dirs if path.exists()):
        print("--- Внимание: cublas64_12.dll не найден в известных DLL-папках. ---")
        print(f"--- Положите cuBLAS/cuDNN DLL в: {LOCAL_CUDA_DLL_DIR} ---")

    return registered_dirs


def find_dll_path(dll_name: str, search_dirs: List[Path]) -> Path | None:
    for search_dir in search_dirs:
        candidate = search_dir / dll_name
        if candidate.exists():
            return candidate
    return None


def diagnose_cuda_dlls(search_dirs: List[Path]) -> None:
    if sys.platform != "win32":
        return

    print("--- Диагностика CUDA DLL ---")
    missing = []

    for dll_name in REQUIRED_CUDA_DLLS:
        dll_path = find_dll_path(dll_name, search_dirs)
        if not dll_path:
            missing.append(dll_name)
            print(f"[MISSING] {dll_name}")
            continue

        try:
            ctypes.WinDLL(str(dll_path))
            print(f"[OK] {dll_name}: {dll_path}")
        except OSError as exc:
            print(f"[FAIL LOAD] {dll_name}: {dll_path} ({exc})")

    if missing:
        print("--- Не хватает DLL для CUDA. GPU-режим может не запуститься. ---")
        print("--- Скопируйте недостающие файлы в cuda_dll или cuda_dll\\bin. ---")


def build_ydl_options(ffmpeg_path: str) -> dict:
    return {
        "format": "bestaudio/best",
        "ffmpeg_location": ffmpeg_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "outtmpl": TEMP_AUDIO_TEMPLATE,
    }


def prepare_audio_source(source: str, is_url: bool, ydl_opts: dict) -> str:
    if not is_url:
        return source

    if os.path.exists(TEMP_AUDIO_FILE):
        print(f"--- Найден {TEMP_AUDIO_FILE}, использую его без повторного скачивания. ---")
        return TEMP_AUDIO_FILE

    print("--- Скачиваю аудио из ВК... ---")
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([source])
    return TEMP_AUDIO_FILE


def should_delete_temp_audio() -> bool:
    if not sys.stdin or not sys.stdin.isatty():
        print(f"--- Неинтерактивный режим: временный файл сохранен ({TEMP_AUDIO_FILE}) ---")
        return False

    answer = input(f"Удалить временный файл {TEMP_AUDIO_FILE}? [y/N]: ").strip().lower()
    return answer in {"y", "yes", "д", "да"}


def create_whisper_model(prefer_gpu: bool = True) -> Tuple[WhisperModel, str]:
    print("--- Загружаю нейросеть... ---")
    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"--- Локальный каталог модели: {LOCAL_MODEL_DIR} ---")

    if prefer_gpu:
        for compute_type in GPU_COMPUTE_TYPES:
            try:
                model = WhisperModel(
                    MODEL_SIZE,
                    device="cuda",
                    compute_type=compute_type,
                    download_root=str(LOCAL_MODEL_DIR),
                )
                print(f"--- Запуск на GPU: compute_type={compute_type} ---")
                return model, "cuda"
            except Exception as exc:
                print(f"--- CUDA не инициализирована для compute_type={compute_type}: {exc} ---")

    print("--- Перехожу на CPU (compute_type=int8) ---")
    return (
        WhisperModel(
            MODEL_SIZE,
            device="cpu",
            compute_type="int8",
            download_root=str(LOCAL_MODEL_DIR),
        ),
        "cpu",
    )


def is_cuda_runtime_error(exc: Exception) -> bool:
    message = str(exc).lower()
    markers = (
        "cublas",
        "cudnn",
        "cuda",
        "cannot be loaded",
        "is not found",
    )
    return any(marker in message for marker in markers)


def collect_transcript_lines(model: WhisperModel, audio_filename: str) -> List[str]:
    lines = []
    last_percent = -1
    total_duration = None

    # Стараемся взять длительность для вывода процента прогресса.
    segments, info = model.transcribe(
        audio_filename,
        beam_size=DEFAULT_BEAM_SIZE,
        language=DEFAULT_LANGUAGE,
    )
    total_duration = getattr(info, "duration", None) or getattr(info, "duration_after_vad", None)

    for idx, segment in enumerate(segments, start=1):
        timestamp = "[%.2fs -> %.2fs]" % (segment.start, segment.end)
        line = f"{timestamp} {segment.text}\n"
        lines.append(line)

        if total_duration and total_duration > 0:
            percent = min(100, int((segment.end / total_duration) * 100))
            if percent != last_percent:
                print(f"\r--- Транскрибация: {percent:3d}% ({segment.end:.1f}/{total_duration:.1f}s) ---", end="")
                last_percent = percent
        else:
            print(f"\r--- Обработан сегмент: {idx} ---", end="")

    if last_percent >= 0:
        print()
    else:
        print("\r--- Транскрибация завершена ---")

    return lines


def transcribe_with_fallback(audio_filename: str) -> Tuple[List[str], str]:
    model, device = create_whisper_model(prefer_gpu=True)

    try:
        return collect_transcript_lines(model, audio_filename), device
    except RuntimeError as exc:
        if device == "cuda" and is_cuda_runtime_error(exc):
            print(f"--- Ошибка CUDA во время распознавания: {exc} ---")
            print("--- Повторяю распознавание на CPU... ---")
            cpu_model, _ = create_whisper_model(prefer_gpu=False)
            return collect_transcript_lines(cpu_model, audio_filename), "cpu"
        raise

def transcribe_video(source, is_url=True):
    ydl_opts = build_ydl_options(DEFAULT_FFMPEG_PATH)
    audio_filename = prepare_audio_source(source, is_url, ydl_opts)

    try:
        print("--- Начинаю распознавание (это может занять время)... ---")
        lines, used_device = transcribe_with_fallback(audio_filename)

        with open(TRANSCRIPT_FILE, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line)

        print(f"\n--- Готово! Текст сохранен в {TRANSCRIPT_FILE} (device={used_device}) ---")
    finally:
        if is_url and os.path.exists(TEMP_AUDIO_FILE):
            if should_delete_temp_audio():
                os.remove(TEMP_AUDIO_FILE)
                print(f"--- Временный файл удален: {TEMP_AUDIO_FILE} ---")
            else:
                print(f"--- Временный файл сохранен: {TEMP_AUDIO_FILE} ---")


if __name__ == "__main__":
    registered_dirs = register_nvidia_dll_dirs()
    diagnose_cuda_dlls(registered_dirs)

    video_input = "https://vkvideo.ru/video-46417318_456251310?list=ln-FiJzuZJwZtIMhgjAWi"

    if video_input.startswith("http"):
        transcribe_video(video_input, is_url=True)
    else:
        transcribe_video(video_input, is_url=False)