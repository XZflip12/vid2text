import argparse
import ctypes
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, cast

from faster_whisper import WhisperModel
from yt_dlp import YoutubeDL

MODEL_SIZE = "medium"
TRANSCRIPT_FILE = "transcript_old.txt"
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


@dataclass
class RunConfig:
    source: str
    is_url: bool
    output_file: str = TRANSCRIPT_FILE
    ffmpeg_path: str = DEFAULT_FFMPEG_PATH
    language: str = DEFAULT_LANGUAGE
    beam_size: int = DEFAULT_BEAM_SIZE
    model_size: str = MODEL_SIZE
    device_mode: Literal["auto", "cpu"] = "auto"
    temp_audio_policy: Literal["ask", "keep", "delete"] = "ask"


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
    with YoutubeDL(cast(Any, ydl_opts)) as ydl:
        ydl.download([source])
    return TEMP_AUDIO_FILE


def should_delete_temp_audio(policy: Literal["ask", "keep", "delete"]) -> bool:
    if policy == "delete":
        return True
    if policy == "keep":
        return False

    if not sys.stdin or not sys.stdin.isatty():
        print(f"--- Неинтерактивный режим: временный файл сохранен ({TEMP_AUDIO_FILE}) ---")
        return False

    answer = input(f"Удалить временный файл {TEMP_AUDIO_FILE}? [y/N]: ").strip().lower()
    return answer in {"y", "yes", "д", "да"}


def create_whisper_model(model_size: str, prefer_gpu: bool = True) -> Tuple[WhisperModel, str]:
    print("--- Загружаю нейросеть... ---")
    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"--- Локальный каталог модели: {LOCAL_MODEL_DIR} ---")

    if prefer_gpu:
        for compute_type in GPU_COMPUTE_TYPES:
            try:
                model = WhisperModel(
                    model_size,
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
            model_size,
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


def collect_transcript_lines(
    model: WhisperModel,
    audio_filename: str,
    beam_size: int,
    language: str,
) -> List[str]:
    lines = []
    last_percent = -1
    total_duration = None

    # Стараемся взять длительность для вывода процента прогресса.
    segments, info = model.transcribe(
        audio_filename,
        beam_size=beam_size,
        language=language,
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


def transcribe_with_fallback(config: RunConfig, audio_filename: str) -> Tuple[List[str], str]:
    model, device = create_whisper_model(
        model_size=config.model_size,
        prefer_gpu=config.device_mode == "auto",
    )

    try:
        return (
            collect_transcript_lines(
                model,
                audio_filename,
                beam_size=config.beam_size,
                language=config.language,
            ),
            device,
        )
    except RuntimeError as exc:
        if device == "cuda" and is_cuda_runtime_error(exc):
            print(f"--- Ошибка CUDA во время распознавания: {exc} ---")
            print("--- Повторяю распознавание на CPU... ---")
            cpu_model, _ = create_whisper_model(model_size=config.model_size, prefer_gpu=False)
            return (
                collect_transcript_lines(
                    cpu_model,
                    audio_filename,
                    beam_size=config.beam_size,
                    language=config.language,
                ),
                "cpu",
            )
        raise


def transcribe_video(config: RunConfig) -> None:
    ydl_opts = build_ydl_options(config.ffmpeg_path)
    audio_filename = prepare_audio_source(config.source, config.is_url, ydl_opts)

    try:
        print("--- Начинаю распознавание (это может занять время)... ---")
        lines, used_device = transcribe_with_fallback(config, audio_filename)

        with open(config.output_file, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line)

        print(f"\n--- Готово! Текст сохранен в {config.output_file} (device={used_device}) ---")
    finally:
        if config.is_url and os.path.exists(TEMP_AUDIO_FILE):
            if should_delete_temp_audio(config.temp_audio_policy):
                os.remove(TEMP_AUDIO_FILE)
                print(f"--- Временный файл удален: {TEMP_AUDIO_FILE} ---")
            else:
                print(f"--- Временный файл сохранен: {TEMP_AUDIO_FILE} ---")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VK/local audio transcription with faster-whisper")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--url", help="Видео URL для скачивания и транскрибации")
    source_group.add_argument("--file", help="Локальный аудиофайл для транскрибации")

    parser.add_argument("--ffmpeg-path", default=DEFAULT_FFMPEG_PATH, help="Путь к ffmpeg.exe")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, help="Язык аудио (например ru, en)")
    parser.add_argument("--beam-size", type=int, default=DEFAULT_BEAM_SIZE, help="Beam size")
    parser.add_argument("--model-size", default=MODEL_SIZE, help="Размер модели Whisper")
    parser.add_argument("--output", default=TRANSCRIPT_FILE, help="Выходной текстовый файл")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu"),
        default="auto",
        help="Режим устройства: auto (GPU с fallback) или cpu",
    )
    parser.add_argument(
        "--temp-audio",
        choices=("ask", "keep", "delete"),
        default="ask",
        help="Поведение для temp_audio.mp3 после обработки URL",
    )
    parser.add_argument(
        "--menu",
        action="store_true",
        help="Принудительно открыть интерактивное меню",
    )
    return parser.parse_args()


def prompt_choice(title: str, choices: List[Tuple[str, str]]) -> str:
    print(f"\n{title}")
    for key, label in choices:
        print(f"  {key}) {label}")

    while True:
        selected = input("Выберите пункт: ").strip()
        if selected in {key for key, _ in choices}:
            return selected
        print("Некорректный выбор, попробуйте снова.")


def prompt_with_default(label: str, default: str) -> str:
    value = input(f"{label} [{default}]: ").strip()
    return value if value else default


def prompt_int_with_default(label: str, default: int, min_value: int = 1) -> int:
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if not raw:
            return default
        try:
            parsed = int(raw)
            if parsed < min_value:
                raise ValueError
            return parsed
        except ValueError:
            print(f"Введите целое число >= {min_value}.")


def run_interactive_menu() -> Optional[RunConfig]:
    ffmpeg_path = DEFAULT_FFMPEG_PATH
    language = DEFAULT_LANGUAGE
    beam_size = DEFAULT_BEAM_SIZE
    model_size = MODEL_SIZE
    output_file = TRANSCRIPT_FILE
    device_mode: Literal["auto", "cpu"] = "auto"
    temp_audio_policy: Literal["ask", "keep", "delete"] = "ask"

    while True:
        print("\n=== Whisper Transcriber CLI ===")
        print(f"Текущие настройки: lang={language}, beam={beam_size}, model={model_size}, device={device_mode}")
        main_choice = prompt_choice(
            "Главное меню",
            [
                ("1", "Транскрибировать видео по URL"),
                ("2", "Транскрибировать локальный файл"),
                ("3", "Настройки"),
                ("4", "Диагностика CUDA DLL"),
                ("0", "Выход"),
            ],
        )

        if main_choice == "0":
            return None

        if main_choice == "4":
            dirs = register_nvidia_dll_dirs()
            diagnose_cuda_dlls(dirs)
            continue

        if main_choice == "3":
            ffmpeg_path = prompt_with_default("Путь к ffmpeg.exe", ffmpeg_path)
            language = prompt_with_default("Код языка", language)
            beam_size = prompt_int_with_default("Beam size", beam_size)
            model_size = prompt_with_default("Размер модели (tiny/base/small/medium/large-v3)", model_size)
            output_file = prompt_with_default("Выходной файл", output_file)

            device_choice = prompt_choice(
                "Режим устройства",
                [("1", "auto (GPU + fallback CPU)"), ("2", "cpu only")],
            )
            device_mode = "auto" if device_choice == "1" else "cpu"

            temp_choice = prompt_choice(
                "Что делать с temp_audio.mp3 после URL-режима",
                [("1", "Спрашивать каждый раз"), ("2", "Всегда сохранять"), ("3", "Всегда удалять")],
            )
            temp_audio_policy = cast(Literal["ask", "keep", "delete"], {"1": "ask", "2": "keep", "3": "delete"}[temp_choice])
            continue

        if main_choice == "1":
            source = input("Введите URL: ").strip()
            if not source:
                print("URL не может быть пустым.")
                continue
            return RunConfig(
                source=source,
                is_url=True,
                output_file=output_file,
                ffmpeg_path=ffmpeg_path,
                language=language,
                beam_size=beam_size,
                model_size=model_size,
                device_mode=device_mode,
                temp_audio_policy=temp_audio_policy,
            )

        source = input("Введите путь к локальному аудиофайлу: ").strip()
        if not source:
            print("Путь не может быть пустым.")
            continue
        if not Path(source).exists():
            print("Файл не найден.")
            continue
        return RunConfig(
            source=source,
            is_url=False,
            output_file=output_file,
            ffmpeg_path=ffmpeg_path,
            language=language,
            beam_size=beam_size,
            model_size=model_size,
            device_mode=device_mode,
            temp_audio_policy=temp_audio_policy,
        )


def build_config_from_args(args: argparse.Namespace) -> Optional[RunConfig]:
    if not args.url and not args.file:
        return None
    source = args.url if args.url else args.file
    return RunConfig(
        source=source,
        is_url=bool(args.url),
        output_file=args.output,
        ffmpeg_path=args.ffmpeg_path,
        language=args.language,
        beam_size=args.beam_size,
        model_size=args.model_size,
        device_mode=args.device,
        temp_audio_policy=args.temp_audio,
    )


def main() -> None:
    args = parse_args()

    config = build_config_from_args(args)
    if args.menu or config is None:
        config = run_interactive_menu()

    if config is None:
        print("Выход.")
        return

    if config.device_mode == "auto":
        registered_dirs = register_nvidia_dll_dirs()
        diagnose_cuda_dlls(registered_dirs)

    transcribe_video(config)


if __name__ == "__main__":
    main()
