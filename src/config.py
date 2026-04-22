from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FAISS_DIR = BASE_DIR / "faiss_index"
RESULTS_DIR = BASE_DIR / "results"
TESTS_DIR = BASE_DIR / "tests"

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

TOP_K = 3