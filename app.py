import os
from pathlib import Path
from typing import List, Tuple, cast, Literal

from modules.settings import AppConfig, load_settings, save_settings
from modules.downloader import prepare_audio_source, handle_temp_audio_cleanup
from modules.transcriber import diagnose_cuda_dlls, transcribe_with_fallback, register_nvidia_dll_dirs

def prompt_choice(title: str, choices: List[Tuple[str, str]]) -> str:
    print(f"\n{title}")
    for key, label in choices:
        print(f"  {key}) {label}")
    while True:
        selected = input("Выберите пункт: ").strip()
        if selected in {key for key, _ in choices}:
            return selected
        print("Некорректный выбор, попробуйте снова.")

def run_settings_menu(config: AppConfig) -> None:
    print("\n=== Настройки ===")
    
    val = input(f"Путь к ffmpeg.exe [{config.ffmpeg_path}]: ").strip()
    if val: config.ffmpeg_path = val

    val = input(f"Код языка [{config.language}]: ").strip()
    if val: config.language = val

    val = input(f"Beam size [{config.beam_size}]: ").strip()
    if val:
        try: config.beam_size = int(val)
        except ValueError: print("Ошибка: нужно число. Оставлено старое значение.")

    val = input(f"Размер модели [{config.model_size}]: ").strip()
    if val: config.model_size = val

    val = input(f"Выходной файл [{config.output_file}]: ").strip()
    if val: config.output_file = val

    device_choice = prompt_choice("Режим устройства", [("1", "auto (GPU + fallback CPU)"), ("2", "cpu only")])
    config.device_mode = "auto" if device_choice == "1" else "cpu"

    temp_choice = prompt_choice("Что делать с temp_audio.mp3 (URL-режим)", [("1", "Спрашивать"), ("2", "Сохранять"), ("3", "Удалять")])
    config.temp_audio_policy = cast(Literal["ask", "keep", "delete"], {"1": "ask", "2": "keep", "3": "delete"}[temp_choice])

    save_settings(config)
    print("\n--- Настройки успешно сохранены в settings.json ---")

def process_transcription(source: str, is_url: bool, config: AppConfig) -> None:
    if config.device_mode == "auto":
        register_nvidia_dll_dirs()

    audio_filename = prepare_audio_source(source, is_url, config.ffmpeg_path)

    try:
        print("--- Начинаю распознавание (это может занять время)... ---")
        used_device = transcribe_with_fallback(audio_filename, config)
        
        abs_path = Path(config.output_file).resolve()
        print(f"\n--- Готово! Текст сохранен в: {abs_path} (device={used_device}) ---")
        
    finally:
        handle_temp_audio_cleanup(is_url, config.temp_audio_policy)
        
        # Обход краша CTranslate2 при сборке мусора на Windows
        print("\nНажмите Enter для выхода из программы...")
        input()
        os._exit(0) 

def main() -> None:
    config = load_settings()

    while True:
        print("\n=== Whisper Transcriber CLI ===")
        print(f"Текущие настройки: lang={config.language}, beam={config.beam_size}, model={config.model_size}, device={config.device_mode}")
        
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
            print("Выход.")
            break
        elif main_choice == "4":
            diagnose_cuda_dlls()
        elif main_choice == "3":
            run_settings_menu(config)
        elif main_choice == "1":
            source = input("Введите URL: ").strip()
            if source:
                process_transcription(source, True, config)
                break
        elif main_choice == "2":
            source = input("Введите путь к локальному аудиофайлу: ").strip()
            if source and Path(source).exists():
                process_transcription(source, False, config)
                break
            else:
                print("Файл не найден или путь пуст.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрограмма прервана пользователем.")
        os._exit(0)
