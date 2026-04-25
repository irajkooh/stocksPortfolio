import logging
from core.config import CHROMA_DIR
from core.runtime import DEVICE

logger = logging.getLogger(__name__)

_vectorstore = None


def _embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        from langchain_chroma import Chroma
        _vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=_embeddings(),
            collection_name="portfolio_kb",
        )
    return _vectorstore


def add_documents(texts: list[str], metadatas: list[dict] | None = None) -> None:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
    docs = []
    for i, text in enumerate(texts):
        meta = (metadatas[i] if metadatas else {}) or {}
        for chunk in splitter.split_text(text):
            docs.append(Document(page_content=chunk, metadata=meta))

    get_vectorstore().add_documents(docs)
    logger.info("Added %d chunks to knowledge base", len(docs))


def query_kb(question: str, k: int = 5) -> list[str]:
    try:
        results = get_vectorstore().similarity_search(question, k=k)
        return [r.page_content for r in results]
    except Exception:
        return []


def kb_size() -> int:
    try:
        return get_vectorstore()._collection.count()
    except Exception:
        return 0
