from app.settings import loadusersettings, registernvidiadlldirs, saveusersettings, diagnosecudadlls
from app.translator import RunConfig
from app.validator import Validator


class Menu:
    def __init__(self, validator: Validator):
        self.validator = validator
        self.settings = loadusersettings()

    def promptchoice(self, title: str, choices: list[tuple[str, str]]) -> str:
        print(title)
        for key, label in choices:
            print(key, label)
        allowed = {k for k, _ in choices}
        while True:
            selected = input("Select an option: ").strip()
            if selected in allowed:
                return selected
            print("Invalid choice, please try again.")

    def promptwithdefault(self, label: str, default: str) -> str:
        value = input(f"{label} [{default}]: ").strip()
        return value if value else default

    def promptintwithdefault(self, label: str, default: int, minvalue: int = 1) -> int:
        while True:
            raw = input(f"{label} [{default}]: ").strip()
            if not raw:
                return default
            try:
                parsed = int(raw)
                if parsed < minvalue:
                    raise ValueError
                return parsed
            except ValueError:
                print(f"Enter an integer >= {minvalue}.")

    def runinteractive(self) -> RunConfig | None:
        while True:
            print("Whisper Transcriber CLI")
            mainchoice = self.promptchoice(
                "Main menu",
                [("1", "Transcribe from URL"), ("2", "Transcribe local file"), ("3", "Settings"), ("4", "Run CUDA DLL diagnostics"), ("0", "Exit")],
            )
            if mainchoice == "0":
                return None
            if mainchoice == "4":
                dirs = registernvidiadlldirs()
                diagnosecudadlls(dirs)
                continue
            if mainchoice == "3":
                self.settings["ffmpegpath"] = self.promptwithdefault("Path to ffmpeg.exe", self.settings["ffmpegpath"])
                self.settings["language"] = self.promptwithdefault("Language code", self.settings["language"])
                self.settings["beamsize"] = self.promptintwithdefault("Beam size", self.settings["beamsize"])
                self.settings["modelsize"] = self.promptwithdefault("Model size tiny/base/small/medium/large-v3", self.settings["modelsize"])
                self.settings["outputfile"] = self.promptwithdefault("Output file", self.settings["outputfile"])
                self.settings["devicemode"] = "auto" if self.promptchoice("Device mode", [("1", "auto GPU fallback CPU"), ("2", "cpu only")]) == "1" else "cpu"
                self.settings["tempaudiopolicy"] = {"1": "ask", "2": "keep", "3": "delete"}[self.promptchoice("Temp audio", [("1", "ask"), ("2", "keep"), ("3", "delete")])]
                saveusersettings(self.settings)
                continue
            source = input("Enter source URL: " if mainchoice == "1" else "Enter local audio file path: ").strip()
            try:
                if mainchoice == "1":
                    source = self.validator.validate_nonempty(source, "URL cannot be empty.")
                else:
                    source = self.validator.validate_local_file(source)
            except Exception as e:
                print(e)
                continue
            return RunConfig(
                source=source,
                isurl=(mainchoice == "1"),
                outputfile=self.settings["outputfile"],
                ffmpegpath=self.settings["ffmpegpath"],
                language=self.settings["language"],
                beamsize=self.settings["beamsize"],
                modelsize=self.settings["modelsize"],
                devicemode=self.settings["devicemode"],
                tempaudiopolicy=self.settings["tempaudiopolicy"],
            )
