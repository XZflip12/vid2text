import os
import sys
from typing import Any, cast, Literal
from yt_dlp import YoutubeDL
from modules.settings import TEMP_AUDIO_TEMPLATE, TEMP_AUDIO_FILE

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

def prepare_audio_source(source: str, is_url: bool, ffmpeg_path: str) -> str:
    if not is_url:
        return source

    if os.path.exists(TEMP_AUDIO_FILE):
        print(f"--- Найден {TEMP_AUDIO_FILE}, использую его без повторного скачивания. ---")
        return TEMP_AUDIO_FILE

    print("--- Скачиваю аудио... ---")
    ydl_opts = build_ydl_options(ffmpeg_path)
    with YoutubeDL(cast(Any, ydl_opts)) as ydl:
        ydl.download([source])
    return TEMP_AUDIO_FILE

def handle_temp_audio_cleanup(is_url: bool, policy: Literal["ask", "keep", "delete"]) -> None:
    if not is_url or not os.path.exists(TEMP_AUDIO_FILE):
        return

    should_delete = False
    if policy == "delete":
        should_delete = True
    elif policy == "ask":
        if sys.stdin and sys.stdin.isatty():
            answer = input(f"Удалить временный файл {TEMP_AUDIO_FILE}? [y/N]: ").strip().lower()
            should_delete = answer in {"y", "yes", "д", "да"}
        else:
            print("--- Неинтерактивный режим: временный файл сохранен ---")

    if should_delete:
        try:
            os.remove(TEMP_AUDIO_FILE)
            print(f"--- Временный файл удален: {TEMP_AUDIO_FILE} ---")
        except OSError:
            pass
    else:
        print(f"--- Временный файл сохранен: {TEMP_AUDIO_FILE} ---")
