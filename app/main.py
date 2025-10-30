# app/main.py
from fastapi import FastAPI, Request
from fastapi.responses import Response
from app.api.routers.ingest import router as ingest_router
from app.api.routers.chat import router as chat_router
import os
import httpx


app = FastAPI()

@app.middleware("http") #для вывода в UTF8 кодировке
async def add_utf8_header(request: Request, call_next):
    response: Response = await call_next(request)
    ct = response.headers.get("content-type")
    if ct and ct.startswith("application/json") and "charset" not in ct:
        response.headers["content-type"] = "application/json; charset=utf-8"
    return response

@app.get("/health")
def health():
    return {"status": "ok"}

@app.on_event("startup")
async def warmup():
    if os.getenv("LLM_PROVIDER", "ollama") != "ollama":
        return
    host = os.getenv("OLLAMA_HOST", "http://ollama:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.1")
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=3.0, read=20.0, write=10.0, pool=5.0)) as client:
            # проверим, что модель существует, и прогреем коротким вызовом
            await client.get(f"{host}/api/tags")
            await client.post(f"{host}/api/chat", json={
                "model": model,
                "messages": [{"role": "user", "content": "ping"}],
                "stream": False,
                "options": {"num_ctx": 1024, "temperature": 0.0}
            })
    except Exception as e:
        # Не роняем приложение, просто лог
        print(f"[warmup] Ollama warmup skipped: {e}")

# роуты
app.include_router(ingest_router, prefix="/ingest")
app.include_router(chat_router)
