from fastapi import APIRouter
from pydantic import BaseModel
from services.knowledge_base import query_kb
from services.llm_service import get_llm
from langchain_core.prompts import ChatPromptTemplate

router = APIRouter(prefix="/chat", tags=["chat"])

_PROMPT = ChatPromptTemplate.from_template(
    "You are a portfolio management assistant.\n"
    "Answer ONLY using the context below. "
    'If the answer is not in the context, reply with exactly: "I don\'t know."\n\n'
    "Context:\n{context}\n\n"
    "Question: {question}\n\nAnswer:"
)


class ChatRequest(BaseModel):
    message: str


@router.post("/")
def chat(body: ChatRequest):
    docs = query_kb(body.message)
    if not docs:
        return {"response": "I don't know."}
    context = "\n\n".join(docs)
    llm     = get_llm()
    result  = (_PROMPT | llm).invoke({"context": context, "question": body.message})
    text    = result.content if hasattr(result, "content") else str(result)
    return {"response": text.strip() or "I don't know."}
