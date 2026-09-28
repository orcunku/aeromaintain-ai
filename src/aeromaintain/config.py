from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DOCUMENTS_DIR = DATA_DIR / "documents"

RANDOM_SEED = 42


for directory in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    DOCUMENTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)