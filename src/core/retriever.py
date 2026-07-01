import os

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CHROMA_DIR = os.path.join(_ROOT, "data", "indexes", "chroma")
COLLECTION_NAME = "sages_szkolenia"


def get_retriever():
    if not os.path.isdir(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
        raise RuntimeError(
            f"Indeks Chroma nie istnieje: {CHROMA_DIR}\n"
            "Uruchom najpierw: python scripts/build_index.py"
        )
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vs.as_retriever(search_kwargs={"k": 3})
