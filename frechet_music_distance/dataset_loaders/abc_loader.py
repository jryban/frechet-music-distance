from pathlib import Path
from typing import Union

from .dataset_loader import DatasetLoader


class ABCLoader(DatasetLoader):

    def __init__(self, verbose: bool = True) -> None:
        supported_extensions = (".abc",)
        super().__init__(supported_extensions, verbose)

    def load_file(self, filepath: Union[str, Path]) -> str:
        self._validate_file(filepath)

        with open(filepath, "r", encoding="utf-8") as file:
            data = file.read()

        return data
