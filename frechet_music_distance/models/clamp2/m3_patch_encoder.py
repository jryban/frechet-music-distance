import torch
from transformers import BertConfig, BertModel, PreTrainedModel

from .config import M3_HIDDEN_SIZE, PATCH_SIZE


class M3PatchEncoder(PreTrainedModel):

    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)
        self.patch_embedding = torch.nn.Linear(PATCH_SIZE*128, M3_HIDDEN_SIZE)
        torch.nn.init.normal_(self.patch_embedding.weight, std=0.02)
        self.base = BertModel(config=config)
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.mask_token_id = 3

    def forward(
        self,
        input_patches: torch.Tensor, # [batch_size, seq_length, hidden_size]
        input_masks: torch.Tensor    # [batch_size, seq_length]
    ) -> torch.Tensor:

        # Transform input_patches into embeddings
        input_patches = torch.nn.functional.one_hot(input_patches, num_classes=128)
        input_patches = input_patches.reshape(len(input_patches), -1, PATCH_SIZE*128).type(torch.FloatTensor)
        input_patches = self.patch_embedding(input_patches.to(self.device))

        # Apply BERT model to input_patches and input_masks
        return self.base(inputs_embeds=input_patches, attention_mask=input_masks)
