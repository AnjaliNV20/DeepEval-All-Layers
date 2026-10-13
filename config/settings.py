import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Values in a local .env file (see .env.example) override the defaults below.
load_dotenv(BASE_DIR / ".env")

# ---------------- Paths ----------------
DOCUMENT_PATH = BASE_DIR / "documents" / "ecommerce.txt"

VECTOR_DB_PATH = BASE_DIR / "vector_db"
REPORT_FILE = Path("reports/evaluation_report.json")

# ---------------- Test data (one folder per suite) ----------------
TESTDATA_DIR = BASE_DIR / "testdata"

# RAG layer
RETRIEVAL_DATASET = (BASE_DIR/ "testdata"/ "rag"/ "rag_retrieval_dataset.json")

GROUNDING_DATASET = (BASE_DIR/ "testdata"/ "rag"/ "rag_generation_dataset.json")

# LLM / UI / safety tests (test_data.json)
LLM_UI_SAFETY_DATASET = TESTDATA_DIR / "llm_ui_safety" / "test_data.json"

# JSON-driven suites (test cases keyed by id)
SINGLE_TURN_DATASET = TESTDATA_DIR / "single_turn_ui_llm" / "single_turn_ui_llm.json"
MULTI_TURN_DATASET = TESTDATA_DIR / "multi_turn_conversation" / "multi_turn_conversation.json"
AGENTIC_DATASET = TESTDATA_DIR / "agentic_testing" / "agentic_testing.json"

# ---------------- Ollama ----------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))

# Setting OLLAMA_MODEL in .env switches every suite below to that one model.
# Each suite keeps its own default if it is not set.

# RAG tests
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL") or "qwen2.5:7b"

# LLM / UI / safety tests (judge model for Answer Relevancy, Faithfulness, ...)
UI_LLM_MODEL = (
    os.getenv("OLLAMA_MODEL_NAME")
    or os.getenv("OLLAMA_MODEL")
    or "qwen2.5:3b"
)

# Single-turn, multi-turn and agentic tests: one model answers, another judges
ANSWER_MODEL_NAME = (
    os.getenv("OLLAMA_ANSWER_MODEL_NAME")
    or os.getenv("OLLAMA_MODEL")
    or "qwen2.5:7b"
)
JUDGE_MODEL_NAME = os.getenv("OLLAMA_JUDGE_MODEL_NAME") or "qwen2.5:3b"

# ---------------- Pass mark ----------------
THRESHOLD = 0.7
