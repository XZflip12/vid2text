import json
import os


class Localizer:
    def __init__(self):
        language = self.__get_language()
        path = os.path.join('locales', f'{language}.json')
        try:
            with open(path, 'r', encoding='utf-8') as file:
                self.text = json.load(file)
        except:
            path = os.path.join('locales', 'en.json')
            with open(path, 'r', encoding='utf-8') as file:
                self.text = json.load(file)

    def get_string(self, key: str, *args):
        text = self.text.get(key, key)
        if args:
            return text.format(*args)
        return text

    def refresh_language(self):
        language = self.__get_language()
        path = os.path.join('locales', f'{language}.json')
        with open(path, 'r', encoding='utf-8') as file:
            self.text = json.load(file)

    def __get_language(self):
        path = os.path.join('settings.json')
        with open(path, 'r', encoding='utf-8') as file:
            return json.load(file)['program_language']
