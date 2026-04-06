import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Tuple

from faster_whisper import WhisperModel
from yt_dlp import YoutubeDL

from app.settings import PROJECTROOT

MODELSIZE = "medium"
TRANSCRIPTFILE = "transcript_old.txt"
TEMPAUDIOTEMPLATE = "tempaudio.%(ext)s"
TEMPAUDIOFILE = "tempaudio.mp3"
DEFAULTLANGUAGE = "ru"
DEFAULTBEAMSIZE = 5
DEFAULTFFMPEGPATH = "ffmpeg"
GPUCOMPUTETYPES = ["float16", "int8_float16", "int8"]
LOCALMODELDIR = PROJECTROOT / "models"


@dataclass
class RunConfig:
    source: str
    isurl: bool
    outputfile: str
    ffmpegpath: str
    language: str
    beamsize: int
    modelsize: str
    devicemode: Literal["auto", "cpu"]
    tempaudiopolicy: Literal["ask", "keep", "delete"]


class Translator:
    def __init__(self, config: RunConfig):
        self.config = config

    def buildydloptions(self) -> dict[str, Any]:
        return {
            "format": "bestaudio/best",
            "ffmpeg_location": self.config.ffmpegpath,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "outtmpl": TEMPAUDIOTEMPLATE,
        }

    def prepareaudiosource(self) -> str:
        if not self.config.isurl:
            return self.config.source
        if os.path.exists(TEMPAUDIOFILE):
            return TEMPAUDIOFILE
        with YoutubeDL(self.buildydloptions()) as ydl:
            ydl.download([self.config.source])
        return TEMPAUDIOFILE

    def createwhispermodel(self, prefergpu: bool = True) -> Tuple[WhisperModel, str]:
        LOCALMODELDIR.mkdir(parents=True, exist_ok=True)
        if prefergpu:
            for computetype in GPUCOMPUTETYPES:
                try:
                    return WhisperModel(self.config.modelsize, device="cuda", compute_type=computetype, download_root=str(LOCALMODELDIR)), "cuda"
                except Exception:
                    pass
        return WhisperModel(self.config.modelsize, device="cpu", compute_type="int8", download_root=str(LOCALMODELDIR)), "cpu"

    def iscudaruntimeerror(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return any(m in message for m in ["cublas", "cudnn", "cuda", "cannot be loaded", "is not found"])

    def collecttranscriptlines(self, model: WhisperModel, audiofilename: str) -> list[str]:
        segments, _info = model.transcribe(audiofilename, beam_size=self.config.beamsize, language=self.config.language)
        return [f"[{s.start:.2f}s - {s.end:.2f}s] {s.text}" for s in segments]

    def transcribewithfallback(self, audiofilename: str) -> tuple[list[str], str]:
        model, device = self.createwhispermodel(prefergpu=self.config.devicemode == "auto")
        try:
            return self.collecttranscriptlines(model, audiofilename), device
        except RuntimeError as exc:
            if device == "cuda" and self.iscudaruntimeerror(exc):
                cpumodel, _ = self.createwhispermodel(prefergpu=False)
                return self.collecttranscriptlines(cpumodel, audiofilename), "cpu"
            raise

    def transcribe(self) -> None:
        audiofilename = self.prepareaudiosource()
        try:
            lines, _device = self.transcribewithfallback(audiofilename)
            with open(self.config.outputfile, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
        finally:
            if self.config.isurl and os.path.exists(TEMPAUDIOFILE) and self.config.tempaudiopolicy == "delete":
                os.remove(TEMPAUDIOFILE)
