"""Build or rebuild the Chroma vector index from data/szkolenia.parquet.

Usage:
    python scripts/build_index.py                           # build if index doesn't exist
    REBUILD_INDEX=true python scripts/build_index.py        # force rebuild
"""

import os
import sys
import json
import shutil

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pyarrow.parquet as pq
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from src.core.retriever import CHROMA_DIR, COLLECTION_NAME

PARQUET_PATH = os.path.join(_ROOT, "data", "szkolenia.parquet")


def _load_rows() -> list[dict]:
    if os.path.isfile(PARQUET_PATH):
        rows = pq.read_table(PARQUET_PATH).to_pylist()
        print(f"✅ Dataset wczytany: {len(rows)} szkoleń")
        return rows
    print(f"❌ Nie znaleziono pliku: {PARQUET_PATH}")
    return []


def _chroma_safe_metadata_value(value):
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    try:
        if hasattr(value, "tolist"):
            value = value.tolist()
        return json.dumps(value, ensure_ascii=False) if value else ""
    except (TypeError, ValueError):
        return str(value)


def _rows_to_docs(rows: list[dict]) -> list[Document]:
    docs = []
    for row in rows:
        nazwa = row.get("nazwa", "")
        opis = row.get("opis", "")
        content = f"Szkolenie: {nazwa}\n\n{opis}".strip()
        if not content:
            continue
        metadata = {
            "nazwa": _chroma_safe_metadata_value(nazwa),
            "kategoria": _chroma_safe_metadata_value(row.get("kategoria", "")),
            "dni": _chroma_safe_metadata_value(row.get("dni", "")),
            "pdf_url": _chroma_safe_metadata_value(row.get("pdf_url", "")),
        }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


def build_index():
    if settings.rebuild_index and os.path.isdir(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
        print("🔄 REBUILD_INDEX=true — usunięto stary indeks, budowanie od zera...")

    if os.path.isdir(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print("✅ Indeks już istnieje:", CHROMA_DIR)
        print("   Ustaw REBUILD_INDEX=true w .env, aby przebudować.")
        return

    rows = _load_rows()
    docs = _rows_to_docs(rows)

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=400,
        chunk_overlap=100,
    )
    doc_splits = text_splitter.split_documents(docs) if docs else []

    if not doc_splits:
        doc_splits = [Document(page_content="(brak dokumentów)", metadata={})]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    os.makedirs(CHROMA_DIR, exist_ok=True)
    Chroma.from_documents(
        doc_splits,
        embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    print(f"✅ Indeks Chroma zbudowany: {len(doc_splits)} chunków → {CHROMA_DIR}")


if __name__ == "__main__":
    build_index()
