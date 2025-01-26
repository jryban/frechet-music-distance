from __future__ import annotations

from pathlib import Path

import mido

from .dataset_loader import DatasetLoader


class MIDIasMTFLoader(DatasetLoader):

    def __init__(self, m3_compatible: bool = True, verbose: bool = True) -> None:
        supported_extensions = (".mid", ".midi")
        super().__init__(supported_extensions, verbose)
        self._m3_compatible = m3_compatible

    def load_file(self, filepath: str | Path) -> str:
        self._validate_file(filepath)

        skip_elements = {"text", "copyright", "track_name", "instrument_name",
                        "lyrics", "marker", "cue_marker", "device_name", "sequencer_specific"}
        try:
            # Load a MIDI file
            mid = mido.MidiFile(str(filepath))
            msg_list = ["ticks_per_beat " + str(mid.ticks_per_beat)]

            # Traverse the MIDI file
            for msg in mid.merged_track:
                if not self._m3_compatible or (msg.type != "sysex" and not (msg.is_meta and msg.type in skip_elements)):
                    str_msg = self._msg_to_str(msg)
                    msg_list.append(str_msg)
        except Exception as e:
            msg = f"Could not load file: {filepath}. Error: {e}"
            raise OSError(msg) from e

        return "\n".join(msg_list)

    def _msg_to_str(self, msg: str) -> str:
        str_msg = ""
        for value in msg.dict().values():
            str_msg += " " + str(value)

        return str_msg.strip().encode("unicode_escape").decode("utf-8")
