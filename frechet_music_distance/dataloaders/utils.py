from __future__ import annotations

from pathlib import Path

from .abc_loader import ABCLoader
from .dataloader import DataLoader
from .midi_as_mtf_loader import MIDIasMTFLoader


def get_dataset_ext(dataset_path: str | Path, supported_extensions: set[str] | None = None) -> str | None:
    if supported_extensions is None:
        supported_extensions = {".mid", ".midi", ".abc"}

    for file in Path(dataset_path).rglob("**/*"):
        if file.suffix in supported_extensions:
            return file.suffix
    return None


def get_dataloader_by_extension_and_model(file_ext: str, model_name: str, verbose: bool | None = None) -> DataLoader:
    if model_name == "clamp":
        if file_ext == ".abc":
            return ABCLoader(verbose=verbose)

    elif model_name == "clamp2":
        if file_ext == ".abc":
            return ABCLoader(verbose=verbose)
        elif file_ext in {".midi", ".mid"}:
            return MIDIasMTFLoader(verbose=verbose)

    msg = f"Unsupported file extension {file_ext} and model {model_name} combination"
    raise ValueError(msg)