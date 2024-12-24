import random
import re
from typing import Iterable

from unidecode import unidecode

from .config import PATCH_LENGTH, PATCH_SIZE


class M3Patchilizer:

    def __init__(self) -> None:
        self.delimiters = ["|:", "::", ":|", "[|", "||", "|]", "|"]
        self.regexPattern = "(" + "|".join(map(re.escape, self.delimiters)) + ")"
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.mask_token_id = 3

    def split_bars(self, body: Iterable[str]) -> list[str]:
        bars = re.split(self.regexPattern, "".join(body))
        bars = list(filter(None, bars))  # remove empty strings
        if bars[0] in self.delimiters:
            bars[1] = bars[0] + bars[1]
            bars = bars[1:]
        bars = [bars[i * 2] + bars[i * 2 + 1] for i in range(len(bars) // 2)]
        return bars

    def bar2patch(self, bar: str, patch_size: int = PATCH_SIZE) -> list[int]:
        patch = [self.bos_token_id] + [ord(c) for c in bar] + [self.eos_token_id]
        patch = patch[:patch_size]
        patch += [self.pad_token_id] * (patch_size - len(patch))
        return patch

    def patch2bar(self, patch: list[int]) -> str:
        return "".join(chr(idx) if idx > self.mask_token_id else "" for idx in patch)

    def encode(
        self,
        item: str,
        patch_size: int = PATCH_SIZE,
        add_special_patches: bool = False,
        truncate: bool = False,
        random_truncate: bool = False
    ) -> list[list[int]]:

        item = unidecode(item)
        lines = re.findall(r".*?\n|.*$", item)
        lines = list(filter(None, lines))  # remove empty lines

        patches = []

        if lines[0].split(" ")[0] == "ticks_per_beat":
            patch = ""
            for line in lines:
                if patch.startswith(line.split(" ")[0]) and (len(patch) + len(" ".join(line.split(" ")[1:])) <= patch_size-2):
                    patch = patch[:-1] + "\t" + " ".join(line.split(" ")[1:])
                else:
                    if patch:
                        patches.append(patch)
                    patch = line
            if patch!="":
                patches.append(patch)
        else:
            for line in lines:
                if len(line) > 1 and ((line[0].isalpha() and line[1] == ':') or line.startswith('%%')):
                    patches.append(line)
                else:
                    bars = self.split_bars(line)
                    if bars:
                        bars[-1] += "\n"
                        patches.extend(bars)

        if add_special_patches:
            bos_patch = chr(self.bos_token_id) * patch_size
            eos_patch = chr(self.eos_token_id) * patch_size
            patches = [bos_patch] + patches + [eos_patch]

        if len(patches) > PATCH_LENGTH and truncate:
            choices = ["head", "tail", "middle"]
            choice = random.choice(choices)
            if choice=="head" or random_truncate is False:
                patches = patches[:PATCH_LENGTH]
            elif choice=="tail":
                patches = patches[-PATCH_LENGTH:]
            else:
                start = random.randint(1, len(patches)-PATCH_LENGTH)
                patches = patches[start:start+PATCH_LENGTH]

        patches = [self.bar2patch(patch) for patch in patches]

        return patches

    def decode(self, patches: list[list[int]]) -> str:
        return "".join(self.patch2bar(patch) for patch in patches)
