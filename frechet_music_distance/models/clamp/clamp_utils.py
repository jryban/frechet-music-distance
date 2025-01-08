import re
from typing import Tuple

import torch
from transformers import BertModel, PreTrainedModel, BertConfig
from unidecode import unidecode

# Constants for patch length and number of features in a patch
PATCH_LENGTH = 64
PATCH_FEATURES = 98


class MusicPatchilizer:
    """
    Class for converting music data to patches and vice-versa.

    Attributes:
        delimiters (tuple): A tuple of strings containing the delimiters used for splitting bars.
        regexPattern (str): A regular expression pattern for splitting bars.
        pad_id (int): The id of the padding token.
        mask_id (int): The id of the mask token.
        eos_id (int): The id of the end-of-sequence token.

    Methods:
        split_bars(body): Splits a body of music into individual bars using the delimiters specified in
        `self.delimiters`.
        bar2patch(bar, patch_length): Encodes a single bar as a patch of specified length.
        patch2bar(patch): Converts a patch to a bar string.
        encode(music, music_length, patch_length=PATCH_LENGTH, add_eos_patch=False):
        Encodes the input music string as a list of patches.
        decode(patches): Decodes a sequence of patches into a music score.
    """

    def __init__(self):
        # Delimiters used for splitting bars
        self.delimiters = "|:", "::", ":|", "[|", "||", "|]", "|"
        # Regular expression pattern for splitting bars
        self.regexPattern = "(" + "|".join(map(re.escape, self.delimiters)) + ")"
        # Padding, mask, and end-of-sequence token ids
        self.pad_id = 0
        self.mask_id = 96
        self.eos_id = 97

    def split_bars(self, body: str):
        """
        Splits a body of music into individual bars using the delimiters specified in `self.delimiters`.

        Args:
            body (str): A string containing the body of music to be split into bars.

        Returns:
            list: A list of strings containing the individual bars.
        """
        body = "".join(body)
        bars = re.split(self.regexPattern, body)
        while "" in bars:
            bars.remove("")
        if bars[0] in self.delimiters:
            bars[1] = bars[0] + bars[1]
            bars = bars[1:]
        bars = [bars[i * 2] + bars[i * 2 + 1] for i in range(int(len(bars) / 2))]

        return bars

    def bar2patch(self, bar: str, patch_length: int) -> list[int]:
        """
        Encodes a single bar as a patch of specified length.

        Args:
            bar (str): A string containing the bar to be encoded.
            patch_length (int): An integer indicating the length of the patch to be returned.

        Returns:
            list: A list of integer-encoded musical tokens.
        """
        patch = [self.pad_id] * patch_length

        for i in range(min(patch_length, len(bar))):
            chr = bar[i]
            idx = ord(chr)
            if  32 <= idx < 127:
                patch[i] = idx - 31

        if i + 1 < patch_length:
            patch[i + 1] = self.eos_id

        return patch

    def patch2bar(self, patch: list[int]) -> str:
        """
        Converts a patch to a bar string.

        Args:
            patch (list): A list of integer-encoded musical tokens.

        Returns:
            str: A string containing the decoded bar.
        """
        bar = ""

        for idx in patch:
            if  0 < idx < 96:
                bar += chr(idx + 31)
            else:
                break

        return bar

    def encode(self, music: str, music_length: int, patch_length: int = PATCH_LENGTH, add_eos_patch: bool = False) -> list[list[int]]:
        """
        Encodes the input music string as a list of patches.

        Args:
            music (str): A string containing the music to be encoded.
            music_length (int): An integer indicating the maximum number of patches to be returned.
            patch_length (int): An integer indicating the length of each patch.
            add_eos_patch (bool): A boolean indicating whether to add an extra patch consisting of all EOS tokens
            at the end of the encoded music.

        Returns:
            list: A list of integer-encoded patches.
        """
        # Convert to ASCII and split into lines
        music = unidecode(music)
        lines = music.split("\n")
        try:
            lines.remove("")
        except:  # noqa: E722
            pass

        body = ""
        patches = []

        # Iterate over lines, splitting bars and encoding each one as a patch
        for line in lines:
            # check if the line is a music score line or not
            if len(line) > 1 and ((line[0].isalpha() and line[1] == ":") or line.startswith("%%score")):
                # if the current line is a music score line, encode the previous body as patches
                if body != "":
                    bars = self.split_bars(body)
                    for bar in bars:
                        # encode each bar in the body as a patch and append to the patches list
                        patch = self.bar2patch(bar, patch_length)
                        patches.append(patch)
                    # reset the body variable
                    body = ""
                # encode the current line as a patch and append to the patches list
                patch = self.bar2patch(line, patch_length)
                patches.append(patch)
            else:
                # if the line is not a music score line, append to the body variable
                body += line

        if body != "":
            bars = self.split_bars(body)

            for bar in bars:
                # encode each bar in the body as a patch and append to the patches list
                patch = self.bar2patch(bar, patch_length)
                patches.append(patch)
        # add an extra patch consisting of all EOS tokens, if required
        if add_eos_patch:
            eos_patch = [self.eos_id] * patch_length
            patches = patches + [eos_patch]

        return patches[:music_length]

    def decode(self, patches: list[list[int]]) -> str:
        """
        Decodes a sequence of patches into a music score.

        Args:
            patches (list): A list of integer-encoded patches.

        Returns:
            str: A string containing the decoded music score.
        """
        music = ""
        for patch in patches:
            music += self.patch2bar(patch) + "\n"
        return music


class MusicEncoder(PreTrainedModel):
    """
    MusicEncoder model for encoding music patches into a sequence of hidden states.

    Args:
        config (:obj:`BertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.

    Attributes:
        patch_embedding (:obj:`torch.nn.Linear`): A linear layer to convert the one-hot encoded patches to the hidden
        size of the model.
        enc (:obj:`BertModel`): The BERT model used to encode the patches.
    """

    def __init__(self, config: BertConfig) -> None:
        super(MusicEncoder, self).__init__(config)
        self.patch_embedding = torch.nn.Linear(PATCH_LENGTH * PATCH_FEATURES, config.hidden_size)
        torch.nn.init.normal_(self.patch_embedding.weight, std=0.02)
        self.enc = BertModel(config=config)

    def forward(self, input_musics: torch.LongTensor, music_masks: torch.LongTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Args:
            input_musics (:obj:`torch.LongTensor` of shape :obj:`(batch_size, music_length, patch_length)`):
                Tensor containing the integer-encoded music patches.
            music_masks (:obj:`torch.LongTensor` of shape :obj:`(batch_size, music_length)`):
                Tensor containing the attention masks for the music patches.

        Returns:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration
            (:class:`~transformers.BertConfig`)
            and inputs:
            last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, music_length, hidden_size)`):
                Sequence of hidden-states at the output of the last layer of the model.
        """
        # One-hot encode the input music patches
        input_musics = torch.nn.functional.one_hot(input_musics, num_classes=PATCH_FEATURES)

        # Reshape the input music patches to feed into the linear layer
        input_musics = input_musics.reshape(len(input_musics), -1, PATCH_LENGTH * PATCH_FEATURES).type(
            torch.FloatTensor
)
        # Apply the linear layer to convert the one-hot encoded patches to hidden features
        input_musics = self.patch_embedding(input_musics.to(self.device))

        # Apply the BERT model to encode the music data
        output = self.enc(inputs_embeds=input_musics, attention_mask=music_masks.to(self.device))

        return output
