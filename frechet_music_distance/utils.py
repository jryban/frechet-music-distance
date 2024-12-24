from pathlib import Path
from typing import Union

import mido
import requests
from tqdm import tqdm

KB = 1024
MB = 1024 * KB


def download_file(url: str, destination: Union[str, Path], verbose: bool = True, chunk_size: int = 10 * MB) -> None:
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))

            if verbose:
                progress_bar = tqdm(total=total_size, unit="B", unit_scale=True)

            destination = Path(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with open(destination, "wb") as file:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if verbose:
                        progress_bar.update(len(chunk))
                    file.write(chunk)

            if verbose:
                progress_bar.close()

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file from url: {url}. Error: {e}")


def load_midi_task(filepath: Union[str, Path], m3_compatible: bool = True) -> str:
    skip_elements = {"text", "copyright", "track_name", "instrument_name",
                     "lyrics", "marker", "cue_marker", "device_name", "sequencer_specific"}
    try:
        # Load a MIDI file
        mid = mido.MidiFile(str(filepath))
        msg_list = ["ticks_per_beat " + str(mid.ticks_per_beat)]

        # Traverse the MIDI file
        for msg in mid.merged_track:
            if not m3_compatible or (msg.type != "sysex" and not (msg.is_meta and msg.type in skip_elements)):
                str_msg = _msg_to_str(msg)
                msg_list.append(str_msg)
    except Exception as e:
        msg = f"Could not load file: {filepath}"
        raise OSError(msg) from e

    return "\n".join(msg_list)


def _msg_to_str(msg: str) -> str:
    str_msg = ""
    for value in msg.dict().values():
        str_msg += " " + str(value)

    return str_msg.strip().encode("unicode_escape").decode("utf-8")


def load_abc_task(filepath: Union[str, Path]) -> str:
    with open(filepath, "r", encoding="utf-8") as file:
        data = file.read()

    return data
