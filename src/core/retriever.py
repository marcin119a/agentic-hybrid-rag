import os

import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CHROMA_DIR = os.path.join(_ROOT, "data", "indexes", "chroma")
COLLECTION_NAME = "sages_szkolenia"

# Gdy ustawione (np. w docker-compose), łączymy się z serwerem Chroma po HTTP
# zamiast otwierać lokalny plik indeksu — pozwala dzielić jeden indeks między
# wieloma kontenerami/procesami.
CHROMA_HOST = os.environ.get("CHROMA_HOST")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))


def chroma_client_kwargs() -> dict:
    if CHROMA_HOST:
        return {"client": chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)}
    return {"persist_directory": CHROMA_DIR}


def get_retriever():
    if not CHROMA_HOST and (not os.path.isdir(CHROMA_DIR) or not os.listdir(CHROMA_DIR)):
        raise RuntimeError(
            f"Indeks Chroma nie istnieje: {CHROMA_DIR}\n"
            "Uruchom najpierw: python scripts/build_index.py"
        )
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        **chroma_client_kwargs(),
    )
    return vs.as_retriever(search_kwargs={"k": 3})
