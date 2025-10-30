from typing import Iterable, List, Dict
from bs4 import BeautifulSoup
from pypdf import PdfReader

# простое разбиение на чанки
def split_text(text: str, max_len: int = 1000, overlap: int = 100) -> list[str]:
    chunks, i = [], 0
    while i < len(text):
        end = min(len(text), i + max_len)
        chunk = text[i:end]
        chunks.append(chunk.strip())
        i = end - overlap
        if i < 0:
            i = end
    return [c for c in chunks if len(c) > 50]

# используем для текста txt или md 
def parse_plain(content: str) -> str:
    return content

def parse_html(content: str) -> str:
    soup = BeautifulSoup(content, "html.parser")
    return soup.get_text("\n")

def parse_pdf(raw: bytes) -> str:
    from io import BytesIO
    pdf = PdfReader(BytesIO(raw))
    text = []
    for page in pdf.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text)
# формируем структуру данных, готовую для векторизации
def make_records(text: str, source: str) -> List[Dict]:
    return [{"source": source, "text": t} for t in split_text(text)]
