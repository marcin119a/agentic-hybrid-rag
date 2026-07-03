from langchain_core.prompts import PromptTemplate
from langchain_core.tools import create_retriever_tool
from langchain_core.tools import tool
import os

from core.hybrid_retriever import (
    get_hybrid_retriever,
    get_faiss_hybrid_retriever,
    FAISS_INDEX_DIR,
)

DOCUMENT_PROMPT = PromptTemplate.from_template(
    "{page_content}\nLink do programu szkolenia: {pdf_url}"
)


retriever = get_hybrid_retriever()
retriever_tool = create_retriever_tool(
    retriever,
    "search_sages_trainings",
    "Wyszukuje szkolenia Sages — zwraca opisy, kategorie, czas trwania i linki do programów.",
    document_prompt=DOCUMENT_PROMPT,
)


@tool
def search_faiss(query: str, k: int = 5) -> str:
    """Wyszukuje najbardziej podobne fragmenty w lokalnym indeksie FAISS.

    Łączy dense (open-source'owy model embeddingowy z HuggingFace, bez klucza
    API OpenAI) i sparse (BM25) ranking przez EnsembleRetriever — BM25 wyłapuje
    dosłowne dopasowania rzadkich nazw własnych, które dense-only wyszukiwanie
    gubi w top-k. Indeks budowany jest przez `scripts/build_faiss_index_demo.py`
    z plików markdown w `data/programy/`.

    Args:
        query: Zapytanie w języku naturalnym (np. "szkolenie z pythona").
        k: Liczba wyników do zwrócenia (domyślnie 5).
    """
    if not os.path.isdir(FAISS_INDEX_DIR) or not os.listdir(FAISS_INDEX_DIR):
        return (
            f"Indeks FAISS nie istnieje: {FAISS_INDEX_DIR}\n"
            "Uruchom najpierw: python scripts/build_faiss_index_demo.py"
        )

    retriever = get_faiss_hybrid_retriever(k=k)
    results = retriever.invoke(query)[:k]
    if not results:
        return "Brak wyników."

    chunks = []
    for doc in results:
        nazwa = doc.metadata.get("nazwa", "")
        kategoria = doc.metadata.get("kategoria", "")
        pdf_url = doc.metadata.get("pdf_url", "")
        chunks.append(
            f"Szkolenie: {nazwa} ({kategoria})\n"
            f"{doc.page_content}\n"
            f"Link do programu szkolenia: {pdf_url}"
        )
    return "\n\n---\n\n".join(chunks)
