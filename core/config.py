import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent

# HF Space writes runtime files to the app dir; local uses ./data
IS_HF_SPACE = os.environ.get("SPACE_ID") is not None
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# LLM
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL    = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
OLLAMA_URL    = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL  = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

# Storage
DB_PATH       = DATA_DIR / "portfolio.db"
DATABASE_URL  = f"sqlite:///{DB_PATH}"
CHROMA_DIR    = DATA_DIR / "chroma_db"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Servers
API_HOST      = "0.0.0.0"
API_PORT      = int(os.environ.get("API_PORT", 8000))
GRADIO_PORT   = int(os.environ.get("GRADIO_PORT", 7860))
