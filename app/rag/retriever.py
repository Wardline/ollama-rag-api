# app/rag/retriever.py

from __future__ import annotations
import os
import json
from typing import List, Sequence, Optional
import numpy as np
import faiss

class Retriever:
    def __init__(
        self,
        backend: str,
        model: str,
        *,
        # НОВОЕ: принимаем index_dir, опционально
        index_dir: Optional[str] = None,
        # Сохраняем обратную совместимость: можно явно передать пути
        faiss_path: Optional[str] = None,
        meta_path: Optional[str] = None,
        ollama_host: Optional[str] = None,
    ) -> None:
        """
        backend: 'ollama' или 'sbert'
        model:   имя эмбеддер-модели
        index_dir: каталог для хранения индекса и метаданных (index.faiss, meta.json)
        faiss_path/meta_path: пути к файлам индекса/метаданных (перекрывают index_dir)
        """
        self.backend = backend
        self.model = model
        self.ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "http://ollama:11434")

        #пути к индексам
        if faiss_path or meta_path:
            # если пути явно заданы, то используем их
            self.faiss_path = faiss_path or os.path.join(index_dir or "data", "index.faiss")
            self.meta_path = meta_path or os.path.join(index_dir or "data", "meta.json")
        else:
            # иначе строим по index_dir
            base = index_dir or "data"
            self.faiss_path = os.path.join(base, "index.faiss")
            self.meta_path = os.path.join(base, "meta.json")

        os.makedirs(os.path.dirname(self.faiss_path), exist_ok=True)

        # инициализируем FAISS и метаданные

        self.dim = None  # установим после первого вычисления эмбеддингов
        if os.path.exists(self.faiss_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.faiss_path)
            with open(self.meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.dim = meta.get("dim")
            self.docs = meta.get("docs", [])

            # в зависимотсти от выбранной модели
            # если docs были списком словарей, извлекаем текст
            if self.docs and isinstance(self.docs[0], dict):
                self.docs = [d.get("text", "") for d in self.docs]

            # если dim в метаданных 0/None, берём из индекса
            if self.dim in (0, None) and self.index is not None:
                # для flat-индекса FAISS размерность можно получить так:
                self.dim = self.index.d
        else:
            # если пустой индекс, то создадим когда узнаем dim
            self.index = None
            self.docs = []

        # Инициализация эмбеддера (как у тебя было)
        # если backend == "ollama": будем бить в /api/embeddings
        # если sbert: грузим SentenceTransformer
        if self.backend == "sbert":
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer(model)
        else:
            self._sbert = None  # ollama

    # два пути получения получения эмбеддингов
    # через Ollama /api/embeddings (альтернатвный)
    def _embed_ollama(self, texts: Sequence[str]) -> np.ndarray:
        import httpx

        timeout = httpx.Timeout(connect=5.0, read=30.0, write=20.0, pool=5.0)
        vecs: list[np.ndarray] = []
        with httpx.Client(timeout=timeout) as client:
            for t in texts:
                model = self.model
                # если забыли тег – подставим :latest для надёжности
                if ":" not in model:
                    model = f"{model}:latest"
                payload = {"model": model, "prompt": t}
                r = client.post(f"{self.ollama_host}/api/embeddings", json=payload)
                if r.status_code != 200:
                    raise RuntimeError(
                        f"Ollama /api/embeddings HTTP {r.status_code}: {r.text}"
                    )
                data = r.json()
                emb = data.get("embedding")
                if emb is None:
                    # на всякий случай поддержка альтернативного формата
                    maybe = data.get("data")
                    if isinstance(maybe, list) and maybe and "embedding" in maybe[0]:
                        emb = maybe[0]["embedding"]
                if emb is None:
                    raise RuntimeError(f"Ollama embeddings payload has no 'embedding' key: {data}")

                vecs.append(np.array(emb, dtype="float32"))

        arr = np.vstack(vecs)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        arr = arr / norms
        return arr.astype("float32")

    # через SBERT (основной)
    def _embed_sbert(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self._sbert.encode(list(texts), normalize_embeddings=True)
        return vecs.astype("float32")

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed_ollama(texts) if self.backend == "ollama" else self._embed_sbert(texts)

    # индексация, добавление новых документов в хранилище FAISS
    def add_texts(self, texts: Sequence[str]) -> None:
        # преобразование текстов в эмбеддинги
        vecs = self.embed(texts) 
        # инициализация индекса
        if self.dim is None:
            self.dim = int(vecs.shape[1])
            self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vecs)
        self.docs.extend(list(texts))
        # сохраняем
        faiss.write_index(self.index, self.faiss_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump({"dim": self.dim, "docs": self.docs}, f, ensure_ascii=False)

    # поиск и извлечение похожих текстов по запросу
    def search(self, query: str, k: int = 3) -> list[str]:
        if self.index is None or self.index.ntotal == 0:
            return []
        q = self.embed([query])
        sims, idxs = self.index.search(q, k)
        res: list[str] = []
        for i in idxs[0]:
            if 0 <= i < len(self.docs):
                res.append(self.docs[i])
        return res
