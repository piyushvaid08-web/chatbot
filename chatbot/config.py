"""Central configuration, loaded from environment variables (.env file)."""
import os

from dotenv import load_dotenv

load_dotenv()

# --- LLM settings -----------------------------------------------------------
# Anything OpenAI-compatible works (OpenAI, DeepSeek, Groq, Gemini, ...).
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip() or ""
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip() or "https://api.openai.com/v1"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "").strip() or "gpt-4o-mini"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip()
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "700"))

# --- Document processing -----------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))          # chars per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))    # chars of overlap
TOP_K = int(os.getenv("TOP_K", "6"))                      # chunks retrieved per question
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "20"))         # turns of chat memory

# --- Paths -------------------------------------------------------------------
DATA_DIR = os.getenv("DATA_DIR", "data").strip()
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")

# Serverless platforms (Vercel etc.) have a read-only app filesystem; only
# /tmp is writable. Fall back so uploads don't crash the deployment.
try:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
except OSError:
    import tempfile

    UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "nova-uploads")
    os.makedirs(UPLOAD_DIR, exist_ok=True)

SUPPORTED_EXTENSIONS = {
    ".xls",
    ".xlsx",
    ".csv",
    ".txt",
    ".md",
    ".log",
    ".pdf",
    ".docx",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
}

# Loading images at startup burns API tokens for zero benefit — they're a
# user-driven upload feature. CLI/web startup only loads these when the user
# passes explicit paths.
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

# Friendly name of the assistant, used in the system prompt and UI.
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "Nova").strip()