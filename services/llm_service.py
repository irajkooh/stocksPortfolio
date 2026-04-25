from core.config import IS_HF_SPACE, GROQ_API_KEY, GROQ_MODEL, OLLAMA_URL, OLLAMA_MODEL


def get_llm():
    """Return Groq (Space/key present) or Ollama (local)."""
    if IS_HF_SPACE or GROQ_API_KEY:
        from langchain_groq import ChatGroq
        return ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0.1)
    else:
        from langchain_community.llms import Ollama
        return Ollama(base_url=OLLAMA_URL, model=OLLAMA_MODEL, temperature=0.1)


def llm_display_name() -> str:
    if IS_HF_SPACE or GROQ_API_KEY:
        return f"Groq · {GROQ_MODEL}"
    return f"Ollama · {OLLAMA_MODEL}"
