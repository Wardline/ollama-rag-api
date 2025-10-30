# app/api/routers/chat.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.core.config import settings
from app.rag.retriever import Retriever


router = APIRouter()

class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    context_used: List[Dict[str, Optional[str]]]

# глобальный экземпляр ретривера (инициализируется лениво)
_retriever: Optional[Retriever] = None

# глобальный экземпляр Retriever
def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever(
            backend=settings.embedding_backend,
            model=settings.embedding_model,
            index_dir=settings.rag_index_path,
        )
    return _retriever

# приводит результаты поиска к единому формату
def normalize_hits(hits: Any) -> List[Dict[str, Optional[str]]]:
    norm: List[Dict[str, Optional[str]]] = []
    if not hits:
        return norm

    for h in hits:
        if isinstance(h, str):
            norm.append({"text": h, "source": None, "chunk_id": None})
        elif isinstance(h, dict):
            text = h.get("text") or h.get("page_content") or ""
            norm.append(
                {
                    "text": text,
                    "source": h.get("source"),
                    "chunk_id": h.get("chunk_id"),
                }
            )
        else:
            # на всякий случай — приводим к строке
            norm.append({"text": str(h), "source": None, "chunk_id": None})
    return norm

# формирует текстовый блок контекста из найденных фрагментов с ограничением длины
def build_context_block(ctx: List[Dict[str, Optional[str]]], max_chars: int = 6000) -> str:

    parts: List[str] = []
    total = 0
    for item in ctx:
        chunk = (item.get("text") or "").strip()
        if not chunk:
            continue
        # обрежем слишком длинные куски
        if len(chunk) > 2000:
            chunk = chunk[:2000] + "..."
        # учитываем разделитель при подсчёте
        add_len = len(chunk) + (4 if parts else 0)
        if total + add_len > max_chars:
            break
        parts.append(chunk)
        total += add_len
    return "\n\n---\n".join(parts)

# отправляет сообщения в Ollama API и возвращает текст ответа модели
async def call_ollama(messages: List[Dict[str, str]]) -> str:
    url = f"{settings.ollama_host}/api/chat"
    payload = {
        "model": getattr(settings, "ollama_model", "llama3.1"),
        "messages": messages,
        "stream": False,
        "options": {
            "num_predict": 256,
            "temperature": 0.2,
            "num_ctx": 4096
        },
        "keep_alive": "5m"
    }

    timeout = httpx.Timeout(connect=5.0, read=180.0, write=60.0, pool=120.0)
    headers = {"Content-Type": "application/json; charset=utf-8"}

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
    except httpx.ReadTimeout:
        raise HTTPException(status_code=504, detail="LLM read timeout")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"LLM HTTP error: {e!s}")

    # Ollama иногда кладёт ответ в data["message"]["content"], иногда — в data["content"]
    msg = (data or {}).get("message") or {}
    answer = msg.get("content") or (data or {}).get("content") or ""
    return answer.strip()




# основной обработчик чата, выполняет поиск по базе знаний и обращается к LLM
@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query")

    # RAG поиск контекста
    retriever = get_retriever()
    k_top = getattr(settings, "k_top", 4)
    raw_hits = retriever.search(query, k=k_top)
    contexts = normalize_hits(raw_hits)

    # Строим промпт
    context_block = build_context_block(contexts)

    user_prompt = (
        f"Вопрос пользователя:\n{query}\n\n"
        + (f"Доступные фрагменты документации:\n{context_block}" if context_block else "Дополнительных фрагментов нет.")
        + "\n\nСформулируй полный и самодостаточный ответ по-русски. "
          "Если в фрагментах есть команды, файлы, пути или конфиги — приведи их дословно. "
          "Не пиши 'см. документацию' и 'см. базу знаний' вместо ответа."
    )

    messages = [
        {
            "role": "system",
            "content": (
                "Ты технический ассистент по настройке и эксплуатации NGFW, Linux и сетевых сервисов. "
                "Твоя задача — выдать готовое решение сразу здесь. "
                "Используй фрагменты документации как источник фактов. "
                "Запрещено отправлять пользователя к 'базе знаний', 'разделу документации' или 'см. выше', "
                "если можно прямо сейчас перечислить шаги. "
                "Если информации явно не хватает — укажи, что не хватает, и предложи разумные варианты."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    # LLM
    answer = await call_ollama(messages)

    # возврат ответа с корректной кодировкой
    payload = {"answer": answer, "context_used": contexts}
    return JSONResponse(content=payload, media_type="application/json; charset=utf-8")

