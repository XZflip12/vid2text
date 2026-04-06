import argparse

from app.menu import Menu
from app.settings import loadusersettings, saveusersettings
from app.translator import Translator
from app.validator import Validator


def parseargs(settings: dict) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="URL/local audio transcription powered by yt-dlp and faster-whisper")
    sourcegroup = parser.add_mutually_exclusive_group()
    sourcegroup.add_argument("--url", help="Source URL to download and transcribe")
    sourcegroup.add_argument("--file", help="Local audio file to transcribe")
    parser.add_argument("--ffmpeg-path", default=settings["ffmpegpath"])
    parser.add_argument("--language", default=settings["language"])
    parser.add_argument("--beam-size", type=int, default=settings["beamsize"])
    parser.add_argument("--model-size", default=settings["modelsize"])
    parser.add_argument("--output", default=settings["outputfile"])
    parser.add_argument("--device", choices=["auto", "cpu"], default=settings["devicemode"])
    parser.add_argument("--temp-audio", choices=["ask", "keep", "delete"], default=settings["tempaudiopolicy"])
    parser.add_argument("--menu", action="store_true")
    return parser.parse_args()


def main() -> None:
    settings = loadusersettings()
    args = parseargs(settings)
    settings.update({
        "ffmpegpath": args.ffmpeg_path,
        "language": args.language,
        "beamsize": args.beam_size,
        "modelsize": args.model_size,
        "outputfile": args.output,
        "devicemode": args.device,
        "tempaudiopolicy": args.temp_audio,
    })
    saveusersettings(settings)
    validator = Validator()
    menu = Menu(validator)
    config = validator.buildconfigfromargs(args)
    if args.menu or config is None:
        config = menu.runinteractive()
        if config is None:
            return
    translator = Translator(config)
    translator.transcribe()


if __name__ == "__main__":
    main()
