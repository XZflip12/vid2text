import os
from pathlib import Path
from typing import List, Tuple, cast, Literal

from modules.settings import AppConfig, load_settings, save_settings
from modules.downloader import prepare_audio_source, handle_temp_audio_cleanup
from modules.transcriber import (diagnose_cuda_dlls, transcribe_with_fallback,
                                 register_nvidia_dll_dirs)
from modules.localizer import Localizer


localizer = Localizer()


def prompt_choice(title: str, choices: List[Tuple[str, str]]) -> str:
    print(f"\n{title}")
    for key, label in choices:
        print(f"  {key}) {label}")
    while True:
        selected = input(localizer.get_string('prompt_select')).strip()
        if selected in {key for key, _ in choices}:
            return selected
        print(localizer.get_string('invalid_choice'))
        print("Некорректный выбор, попробуйте снова.")

def run_settings_menu(config: AppConfig) -> None:
    print(localizer.get_string('menu_settings_title'))

    val = input(localizer.get_string('input_prog_lang', config.program_language)).strip()
    if val: config.program_language = val

    val = input(localizer.get_string('input_ffmpeg_path', config.ffmpeg_path)).strip()
    if val: config.ffmpeg_path = val

    val = input(localizer.get_string('input_whisper_lang', config.language)).strip()
    if val: config.language = val

    val = input(localizer.get_string('input_beam_size', config.beam_size)).strip()
    if val:
        try: config.beam_size = int(val)
        except ValueError: print(localizer.get_string('error_beam_number'))

    val = input(localizer.get_string('input_model_size', config.model_size)).strip()
    if val: config.model_size = val

    val = input(localizer.get_string('input_output_file', config.output_file)).strip()
    if val: config.output_file = val

    device_choice = prompt_choice(
        localizer.get_string('device_mode_title'),
        [("1", "auto (GPU + fallback CPU)"), ("2", "cpu only")])
    config.device_mode = "auto" if device_choice == "1" else "cpu"

    temp_choice = prompt_choice(localizer.get_string('temp_policy_title'),
                                [("1", localizer.get_string('temp_policy_ask')),
                                 ("2", localizer.get_string('temp_policy_keep')),
                                 ("3", localizer.get_string('temp_policy_delete'))])
    config.temp_audio_policy = cast(Literal["ask", "keep", "delete"], {"1": "ask", "2": "keep", "3": "delete"}[temp_choice])

    save_settings(config)
    localizer.refresh_language()
    print(localizer.get_string('settings_saved'))


def process_transcription(source: str, is_url: bool, config: AppConfig) -> None:
    if config.device_mode == "auto":
        register_nvidia_dll_dirs()

    audio_filename = prepare_audio_source(source, is_url, config.ffmpeg_path)

    try:
        print(localizer.get_string('recognition_start'))
        used_device = transcribe_with_fallback(audio_filename, config)
        
        abs_path = Path(config.output_file).resolve()
        print(localizer.get_string('done_success', abs_path, used_device))
        
    finally:
        handle_temp_audio_cleanup(is_url, config.temp_audio_policy)
        
        print(localizer.get_string('press_enter_exit'))
        input()
        os._exit(0) 

def main() -> None:
    config = load_settings()

    while True:
        print(localizer.get_string('menu_app_title'))
        print(localizer.get_string(
            'current_settings_status', config.language, config.beam_size,
            config.model_size, config.device_mode))
        
        main_choice = prompt_choice(
            localizer.get_string('main_menu_label'),
            [
                ("1", localizer.get_string('choice_url')),
                ("2", localizer.get_string('choice_local')),
                ("3", localizer.get_string('choice_settings')),
                ("4", localizer.get_string('choice_diagnostics')),
                ("0", localizer.get_string('choice_exit')),
            ],
        )

        if main_choice == "0":
            print(localizer.get_string('exit_message'))
            break
        elif main_choice == "4":
            diagnose_cuda_dlls()
        elif main_choice == "3":
            run_settings_menu(config)
        elif main_choice == "1":
            source = input(localizer.get_string('input_url')).strip()
            if source:
                process_transcription(source, True, config)
                break
        elif main_choice == "2":
            source = input(localizer.get_string('input_local_path')).strip()
            if source and Path(source).exists():
                process_transcription(source, False, config)
                break
            else:
                print(localizer.get_string('error_file_not_found'))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(localizer.get_string('error_keyboard_interrupt'))
        os._exit(0)
