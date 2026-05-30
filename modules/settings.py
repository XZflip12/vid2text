import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal
from modules.localizer import Localizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_FILE = PROJECT_ROOT / "settings.json"
LOCAL_MODEL_DIR = PROJECT_ROOT / "models"
LOCAL_CUDA_DLL_DIR = PROJECT_ROOT / "cuda_dll"

TRANSCRIPT_FILE = str(PROJECT_ROOT / "transcript_old.txt")
FFMPEG_FILE = str(PROJECT_ROOT / "ffmpeg.exe")
TEMP_AUDIO_TEMPLATE = str(PROJECT_ROOT / "temp_audio.%(ext)s")
TEMP_AUDIO_FILE = str(PROJECT_ROOT / "temp_audio.mp3")

localizer = Localizer()


@dataclass
class AppConfig:
    ffmpeg_path: str = FFMPEG_FILE
    language: str = "en"
    program_language: str = "en"
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
            valid_keys = {k for k in AppConfig.__dataclass_fields__}
            filtered_data = {k: v for k, v in data.items() if k in valid_keys}
            return AppConfig(**filtered_data)
    except Exception as e:
        print(localizer.get_string('error_settings_read', SETTINGS_FILE, e))
        return AppConfig()

def save_settings(config: AppConfig) -> None:
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(asdict(config), f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(localizer.get_string('error_settings_save', e))
