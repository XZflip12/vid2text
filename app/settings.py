import ctypes
import json
import os
import sys
from pathlib import Path
from typing import Any, List

PROJECTROOT = Path(__file__).resolve().parent.parent
LOCALCUDADLLDIR = PROJECTROOT / "cuda_dll"
USERSETTINGSFILE = PROJECTROOT / "usersettings.json"
REQUIREDCUDADLLS = [
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    "cudnn64_9.dll",
    "cudart64_12.dll",
]
DLLDIRHANDLES: List[Any] = []


def defaultusersettings() -> dict[str, Any]:
    return {
        "ffmpegpath": "ffmpeg",
        "language": "ru",
        "beamsize": 5,
        "modelsize": "medium",
        "outputfile": "transcript_old.txt",
        "devicemode": "auto",
        "tempaudiopolicy": "ask",
    }


def sanitizeusersettings(raw: Any) -> dict[str, Any]:
    settings = defaultusersettings()
    if not isinstance(raw, dict):
        return settings
    for key in ["ffmpegpath", "language", "modelsize", "outputfile"]:
        if isinstance(raw.get(key), str) and raw[key].strip():
            settings[key] = raw[key].strip()
    if isinstance(raw.get("beamsize"), int) and raw["beamsize"] >= 1:
        settings["beamsize"] = raw["beamsize"]
    if raw.get("devicemode") in {"auto", "cpu"}:
        settings["devicemode"] = raw["devicemode"]
    if raw.get("tempaudiopolicy") in {"ask", "keep", "delete"}:
        settings["tempaudiopolicy"] = raw["tempaudiopolicy"]
    return settings


def loadusersettings() -> dict[str, Any]:
    if not USERSETTINGSFILE.exists():
        return defaultusersettings()
    try:
        with open(USERSETTINGSFILE, "r", encoding="utf-8") as f:
            return sanitizeusersettings(json.load(f))
    except (OSError, json.JSONDecodeError):
        return defaultusersettings()


def saveusersettings(settings: dict[str, Any]) -> None:
    with open(USERSETTINGSFILE, "w", encoding="utf-8") as f:
        json.dump(sanitizeusersettings(settings), f, ensure_ascii=False, indent=2)


def registernvidiadlldirs() -> list[Path]:
    if sys.platform != "win32":
        return []
    LOCALCUDADLLDIR.mkdir(parents=True, exist_ok=True)
    candidatedirs: list[Path] = [LOCALCUDADLLDIR]
    if (LOCALCUDADLLDIR / "bin").exists():
        candidatedirs.append(LOCALCUDADLLDIR / "bin")
    nvidiaroot = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if nvidiaroot.exists():
        for root, dirs, _ in os.walk(nvidiaroot):
            if "bin" in dirs:
                candidatedirs.append(Path(root) / "bin")
    candidatedirs.extend([Path(v) / "bin" for k, v in os.environ.items() if k.startswith("CUDA_PATH") and v])
    seen, out = set(), []
    for d in candidatedirs:
        if not d.exists():
            continue
        n = str(d.resolve())
        if n in seen:
            continue
        seen.add(n)
        try:
            DLLDIRHANDLES.append(os.add_dll_directory(str(d)))
            os.environ["PATH"] = f"{d}{os.pathsep}" + os.environ.get("PATH", "")
            out.append(d)
        except OSError:
            pass
    return out


def finddllpath(dllname: str, searchdirs: list[Path]) -> Path | None:
    for d in searchdirs:
        p = d / dllname
        if p.exists():
            return p
    return None


def diagnosecudadlls(searchdirs: list[Path]) -> None:
    if sys.platform != "win32":
        return
    for dllname in REQUIREDCUDADLLS:
        dllpath = finddllpath(dllname, searchdirs)
        if dllpath:
            try:
                ctypes.WinDLL(str(dllpath))
            except OSError:
                pass
