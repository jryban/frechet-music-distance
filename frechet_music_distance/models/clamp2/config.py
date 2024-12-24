from pathlib import Path


PATCH_SIZE = 64  # Size of each patch
PATCH_LENGTH = 512  # Length of the patches
PATCH_NUM_LAYERS = 12  # Number of layers in the encoder
TOKEN_NUM_LAYERS = 3  # Number of layers in the decoder
M3_HIDDEN_SIZE = 768  # Size of the hidden layer

# -------------------- Configuration for CLaMP2  ----------------
CLAMP2_HIDDEN_SIZE = 768  # Size of the hidden layer
TEXT_MODEL_NAME = "FacebookAI/xlm-roberta-base"  # Name of the pre-trained text model

CLAMP2_NUM_EPOCH = 100  # Maximum number of epochs for training
CLAMP2_LEARNING_RATE = 5e-5  # Learning rate for the optimizer
CLAMP2_BATCH_SIZE = 128  # Batch size per GPU (single card) during training
LOGIT_SCALE = 1  # Scaling factor for contrastive loss
MAX_TEXT_LENGTH = 128  # Maximum allowed length for text input
TEXT_DROPOUT = True  # Whether to apply dropout during text processing
CLAMP2_DETERMINISTIC = True  # Ensures deterministic results with random seeds
CLAMP2_LOAD_M3 = True  # Load weights from the M3 model
CLAMP2_WEIGHTS_URL = "https://huggingface.co/sander-wood/clamp2/resolve/main/weights_clamp2_h_size_768_lr_5e-05_batch_128_scale_1_t_length_128_t_model_FacebookAI_xlm-roberta-base_t_dropout_True_m3_True.pth"

CLAMP2_WEIGHT_DIR = Path.home() / ".cache" / "frechet_music_distance" / "checkpoints" / "clamp2"
CLAMP_CKPT_NAME = (
    "weights_clamp2_h_size_" + str(CLAMP2_HIDDEN_SIZE) +
    "_lr_" + str(CLAMP2_LEARNING_RATE) +
    "_batch_" + str(CLAMP2_BATCH_SIZE) +
    "_scale_" + str(LOGIT_SCALE) +
    "_t_length_" + str(MAX_TEXT_LENGTH) +
    "_t_model_" + TEXT_MODEL_NAME.replace("/", "_") +
    "_t_dropout_" + str(TEXT_DROPOUT) +
    "_m3_" + str(CLAMP2_LOAD_M3) + ".pth"
)
CLAMP2_WEIGHTS_PATH = CLAMP2_WEIGHT_DIR / CLAMP_CKPT_NAME  # Path to store CLaMP2 model weights
