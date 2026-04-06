from pathlib import Path
from typing import Optional

from app.translator import RunConfig


class Validator:
    def validate_nonempty(self, value: str, message: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError(message)
        return value

    def validate_local_file(self, path: str) -> str:
        path = self.validate_nonempty(path, "Path cannot be empty.")
        if not Path(path).exists():
            raise FileNotFoundError("File not found.")
        return path

    def validate_runconfig(self, config: RunConfig) -> RunConfig:
        if config.beamsize < 1:
            raise ValueError("Beam size must be >= 1.")
        if config.devicemode not in {"auto", "cpu"}:
            raise ValueError("Invalid device mode.")
        if config.tempaudiopolicy not in {"ask", "keep", "delete"}:
            raise ValueError("Invalid temp audio policy.")
        return config

    def buildconfigfromargs(self, args) -> Optional[RunConfig]:
        if not args.url and not args.file:
            return None
        source = args.url if args.url else args.file
        return RunConfig(
            source=source,
            isurl=bool(args.url),
            outputfile=args.output,
            ffmpegpath=args.ffmpegpath,
            language=args.language,
            beamsize=args.beamsize,
            modelsize=args.modelsize,
            devicemode=args.device,
            tempaudiopolicy=args.tempaudio,
        )
