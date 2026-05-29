import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_FILE = PROJECT_ROOT / "settings.json"
LOCAL_MODEL_DIR = PROJECT_ROOT / "models"
LOCAL_CUDA_DLL_DIR = PROJECT_ROOT / "cuda_dll"

# Фиксированные пути для временных файлов
TRANSCRIPT_FILE = str(PROJECT_ROOT / "transcript_old.txt")
TEMP_AUDIO_TEMPLATE = str(PROJECT_ROOT / "temp_audio.%(ext)s")
TEMP_AUDIO_FILE = str(PROJECT_ROOT / "temp_audio.mp3")

@dataclass
class AppConfig:
    ffmpeg_path: str = r"C:\Users\oldfa\OneDrive\Документы\ffmpeg\bin\ffmpeg.exe"
    language: str = "ru"
    beam_size: int = 5
    model_size: str = "medium"
    device_mode: Literal["auto", "cpu"] = "auto"
    temp_audio_policy: Literal["ask", "keep", "delete"] = "ask"
    output_file: str = TRANSCRIPT_FILE

def load_settings() -> AppConfig:
    if not SETTINGS_FILE.exists():
        default_config = AppConfig()
        save_settings(default_config)
        return default_config
    
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Фильтруем ключи, чтобы избежать ошибок, если в JSON есть лишние данные
            valid_keys = {k for k in AppConfig.__dataclass_fields__}
            filtered_data = {k: v for k, v in data.items() if k in valid_keys}
            return AppConfig(**filtered_data)
    except Exception as e:
        print(f"--- Ошибка чтения {SETTINGS_FILE}: {e}. Использую настройки по умолчанию ---")
        return AppConfig()

def save_settings(config: AppConfig) -> None:
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(asdict(config), f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"--- Ошибка сохранения настроек: {e} ---")