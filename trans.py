import argparse
import ctypes
import json
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
DEFAULT_FFMPEG_PATH = "ffmpeg"
GPU_COMPUTE_TYPES = ("float16", "int8_float16", "int8")
PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_MODEL_DIR = PROJECT_ROOT / "models"
LOCAL_CUDA_DLL_DIR = PROJECT_ROOT / "cuda_dll"
USER_SETTINGS_FILE = PROJECT_ROOT / "user_settings.json"
REQUIRED_CUDA_DLLS = (
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    "cudnn64_9.dll",
    "cudart64_12.dll",
)

# On Windows, keep add_dll_directory handles alive so DLL search paths remain active.
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


def default_user_settings() -> dict[str, Any]:
    return {
        "ffmpeg_path": DEFAULT_FFMPEG_PATH,
        "language": DEFAULT_LANGUAGE,
        "beam_size": DEFAULT_BEAM_SIZE,
        "model_size": MODEL_SIZE,
        "output_file": TRANSCRIPT_FILE,
        "device_mode": "auto",
        "temp_audio_policy": "ask",
    }


def sanitize_user_settings(raw: Any) -> dict[str, Any]:
    settings = default_user_settings()
    if not isinstance(raw, dict):
        return settings

    if isinstance(raw.get("ffmpeg_path"), str) and raw["ffmpeg_path"].strip():
        settings["ffmpeg_path"] = raw["ffmpeg_path"].strip()
    if isinstance(raw.get("language"), str) and raw["language"].strip():
        settings["language"] = raw["language"].strip()
    if isinstance(raw.get("model_size"), str) and raw["model_size"].strip():
        settings["model_size"] = raw["model_size"].strip()
    if isinstance(raw.get("output_file"), str) and raw["output_file"].strip():
        settings["output_file"] = raw["output_file"].strip()

    beam_size = raw.get("beam_size")
    if isinstance(beam_size, int) and beam_size >= 1:
        settings["beam_size"] = beam_size

    device_mode = raw.get("device_mode")
    if device_mode in {"auto", "cpu"}:
        settings["device_mode"] = device_mode

    temp_audio_policy = raw.get("temp_audio_policy")
    if temp_audio_policy in {"ask", "keep", "delete"}:
        settings["temp_audio_policy"] = temp_audio_policy

    return settings


def load_user_settings() -> dict[str, Any]:
    if not USER_SETTINGS_FILE.exists():
        return default_user_settings()

    try:
        with open(USER_SETTINGS_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return sanitize_user_settings(raw)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"--- Failed to load {USER_SETTINGS_FILE.name}, using defaults: {exc} ---")
        return default_user_settings()


def save_user_settings(settings: dict[str, Any]) -> None:
    safe_settings = sanitize_user_settings(settings)
    try:
        with open(USER_SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(safe_settings, f, ensure_ascii=True, indent=2)
        print(f"--- Settings saved to {USER_SETTINGS_FILE.name} ---")
    except OSError as exc:
        print(f"--- Failed to save settings: {exc} ---")


def register_nvidia_dll_dirs() -> List[Path]:
    if sys.platform != "win32":
        return []

    LOCAL_CUDA_DLL_DIR.mkdir(parents=True, exist_ok=True)

    candidate_dirs = []

    # Project-local folder for manually supplied cuBLAS/cuDNN DLLs.
    candidate_dirs.append(LOCAL_CUDA_DLL_DIR)
    local_bin_dir = LOCAL_CUDA_DLL_DIR / "bin"
    if local_bin_dir.exists():
        candidate_dirs.append(local_bin_dir)

    # NVIDIA runtime DLLs from the active virtual environment.
    nvidia_root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if nvidia_root.exists():
        for root, dirs, _ in os.walk(nvidia_root):
            if "bin" in dirs:
                candidate_dirs.append(Path(root) / "bin")

    # CUDA Toolkit paths from environment variables (system installation).
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
            print(f"--- Added DLL search path: {dll_dir} ---")
        except OSError as exc:
            print(f"--- Failed to add DLL search path: {dll_dir} ({exc}) ---")

    if registered_count == 0:
        print("--- No DLL paths were added. Check your CUDA runtime folders. ---")

    if not any((path / "cublas64_12.dll").exists() for path in candidate_dirs if path.exists()):
        print("--- Warning: cublas64_12.dll was not found in discovered DLL folders. ---")
        print(f"--- Place cuBLAS/cuDNN DLL files in: {LOCAL_CUDA_DLL_DIR} ---")

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

    print("--- CUDA DLL diagnostics ---")
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
        print("--- Some CUDA DLL files are missing. GPU mode may fail to start. ---")
        print("--- Copy missing files into cuda_dll or cuda_dll\\bin. ---")


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
        print(f"--- Found {TEMP_AUDIO_FILE}, reusing it without re-downloading. ---")
        return TEMP_AUDIO_FILE

    print("--- Downloading audio with yt-dlp... ---")
    with YoutubeDL(cast(Any, ydl_opts)) as ydl:
        ydl.download([source])
    return TEMP_AUDIO_FILE


def should_delete_temp_audio(policy: Literal["ask", "keep", "delete"]) -> bool:
    if policy == "delete":
        return True
    if policy == "keep":
        return False

    if not sys.stdin or not sys.stdin.isatty():
        print(f"--- Non-interactive mode: keeping temporary file ({TEMP_AUDIO_FILE}) ---")
        return False

    answer = input(f"Delete temporary file {TEMP_AUDIO_FILE}? [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def create_whisper_model(model_size: str, prefer_gpu: bool = True) -> Tuple[WhisperModel, str]:
    print("--- Loading transcription model... ---")
    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"--- Local model directory: {LOCAL_MODEL_DIR} ---")

    if prefer_gpu:
        for compute_type in GPU_COMPUTE_TYPES:
            try:
                model = WhisperModel(
                    model_size,
                    device="cuda",
                    compute_type=compute_type,
                    download_root=str(LOCAL_MODEL_DIR),
                )
                print(f"--- Running on GPU: compute_type={compute_type} ---")
                return model, "cuda"
            except Exception as exc:
                print(f"--- CUDA initialization failed for compute_type={compute_type}: {exc} ---")

    print("--- Falling back to CPU (compute_type=int8) ---")
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

    # Use total duration when available to report progress as a percentage.
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
                print(f"\r--- Transcription progress: {percent:3d}% ({segment.end:.1f}/{total_duration:.1f}s) ---", end="")
                last_percent = percent
        else:
            print(f"\r--- Processed segment: {idx} ---", end="")

    if last_percent >= 0:
        print()
    else:
        print("\r--- Transcription completed ---")

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
            print(f"--- CUDA runtime error during transcription: {exc} ---")
            print("--- Retrying on CPU... ---")
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
        print("--- Starting transcription (this may take a while)... ---")
        lines, used_device = transcribe_with_fallback(config, audio_filename)

        with open(config.output_file, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line)

        print(f"\n--- Done! Transcript saved to {config.output_file} (device={used_device}) ---")
    finally:
        if config.is_url and os.path.exists(TEMP_AUDIO_FILE):
            if should_delete_temp_audio(config.temp_audio_policy):
                os.remove(TEMP_AUDIO_FILE)
                print(f"--- Temporary file deleted: {TEMP_AUDIO_FILE} ---")
            else:
                print(f"--- Temporary file kept: {TEMP_AUDIO_FILE} ---")


def parse_args(settings: dict[str, Any]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="URL/local audio transcription powered by yt-dlp and faster-whisper")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--url", help="Source URL to download and transcribe")
    source_group.add_argument("--file", help="Local audio file to transcribe")

    parser.add_argument("--ffmpeg-path", default=settings["ffmpeg_path"], help="Path to ffmpeg.exe")
    parser.add_argument("--language", default=settings["language"], help="Audio language code (e.g. ru, en)")
    parser.add_argument("--beam-size", type=int, default=settings["beam_size"], help="Beam size")
    parser.add_argument("--model-size", default=settings["model_size"], help="Whisper model size")
    parser.add_argument("--output", default=settings["output_file"], help="Output transcript file")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu"),
        default=settings["device_mode"],
        help="Device mode: auto (GPU with CPU fallback) or cpu",
    )
    parser.add_argument(
        "--temp-audio",
        choices=("ask", "keep", "delete"),
        default=settings["temp_audio_policy"],
        help="What to do with temp_audio.mp3 after URL processing",
    )
    parser.add_argument(
        "--menu",
        action="store_true",
        help="Force interactive menu mode",
    )
    return parser.parse_args()


def prompt_choice(title: str, choices: List[Tuple[str, str]]) -> str:
    print(f"\n{title}")
    for key, label in choices:
        print(f"  {key}) {label}")

    while True:
        selected = input("Select an option: ").strip()
        if selected in {key for key, _ in choices}:
            return selected
        print("Invalid choice, please try again.")


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
            print(f"Enter an integer >= {min_value}.")


def run_interactive_menu(settings: dict[str, Any]) -> Optional[RunConfig]:
    ffmpeg_path: str = settings["ffmpeg_path"]
    language: str = settings["language"]
    beam_size: int = settings["beam_size"]
    model_size: str = settings["model_size"]
    output_file: str = settings["output_file"]
    device_mode: Literal["auto", "cpu"] = cast(Literal["auto", "cpu"], settings["device_mode"])
    temp_audio_policy: Literal["ask", "keep", "delete"] = cast(
        Literal["ask", "keep", "delete"], settings["temp_audio_policy"]
    )

    while True:
        print("\n=== Whisper Transcriber CLI ===")
        print(f"Current settings: lang={language}, beam={beam_size}, model={model_size}, device={device_mode}")
        main_choice = prompt_choice(
            "Main menu",
            [
                ("1", "Transcribe from URL"),
                ("2", "Transcribe local file"),
                ("3", "Settings"),
                ("4", "Run CUDA DLL diagnostics"),
                ("0", "Exit"),
            ],
        )

        if main_choice == "0":
            return None

        if main_choice == "4":
            dirs = register_nvidia_dll_dirs()
            diagnose_cuda_dlls(dirs)
            continue

        if main_choice == "3":
            ffmpeg_path = prompt_with_default("Path to ffmpeg.exe", ffmpeg_path)
            language = prompt_with_default("Language code", language)
            beam_size = prompt_int_with_default("Beam size", beam_size)
            model_size = prompt_with_default("Model size (tiny/base/small/medium/large-v3)", model_size)
            output_file = prompt_with_default("Output file", output_file)

            device_choice = prompt_choice(
                "Device mode",
                [("1", "auto (GPU + fallback CPU)"), ("2", "cpu only")],
            )
            device_mode = "auto" if device_choice == "1" else "cpu"

            temp_choice = prompt_choice(
                "What to do with temp_audio.mp3 after URL mode",
                [("1", "Ask every time"), ("2", "Always keep"), ("3", "Always delete")],
            )
            temp_audio_policy = cast(Literal["ask", "keep", "delete"], {"1": "ask", "2": "keep", "3": "delete"}[temp_choice])
            settings.update(
                {
                    "ffmpeg_path": ffmpeg_path,
                    "language": language,
                    "beam_size": beam_size,
                    "model_size": model_size,
                    "output_file": output_file,
                    "device_mode": device_mode,
                    "temp_audio_policy": temp_audio_policy,
                }
            )
            save_user_settings(settings)
            continue

        if main_choice == "1":
            source = input("Enter URL: ").strip()
            if not source:
                print("URL cannot be empty.")
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

        source = input("Enter local audio file path: ").strip()
        if not source:
            print("Path cannot be empty.")
            continue
        if not Path(source).exists():
            print("File not found.")
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
    settings = load_user_settings()
    args = parse_args(settings)

    settings.update(
        {
            "ffmpeg_path": args.ffmpeg_path,
            "language": args.language,
            "beam_size": args.beam_size,
            "model_size": args.model_size,
            "output_file": args.output,
            "device_mode": args.device,
            "temp_audio_policy": args.temp_audio,
        }
    )
    save_user_settings(settings)

    config = build_config_from_args(args)
    if args.menu or config is None:
        config = run_interactive_menu(settings)

    if config is None:
        print("Exit.")
        return

    if config.device_mode == "auto":
        registered_dirs = register_nvidia_dll_dirs()
        diagnose_cuda_dlls(registered_dirs)

    transcribe_video(config)


if __name__ == "__main__":
    main()
