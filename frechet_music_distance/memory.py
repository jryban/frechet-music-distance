from joblib import Memory
from pathlib import Path

CAHE_MEMORY_DIR = Path.home() / ".cache" / "frechet_music_distance" / "precomputed"

MEMORY = Memory(CAHE_MEMORY_DIR, verbose=0)
MEMORY.reduce_size(bytes_limit="10G")
