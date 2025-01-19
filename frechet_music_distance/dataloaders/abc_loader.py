from .dataloader import DataLoader
from typing import Union
from pathlib import Path


class ABCLoader(DataLoader):

    def __init__(self, verbose: bool = True):
        supported_extensions = (".abc",)
        super().__init__(supported_extensions, verbose)

    def load_file(self, filepath: Union[str, Path]) -> str:
        self._validate_file(filepath)

        with open(filepath, "r", encoding="utf-8") as file:
            data = file.read()

        return data
