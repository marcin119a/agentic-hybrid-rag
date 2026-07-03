"""
Przykład budowy indeksu FAISS z wykorzystaniem lokalnego modelu embeddingowego
Hugging Face.

Skrypt wykorzystuje model:
    sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

Embeddingi są generowane lokalnie, bez użycia zewnętrznych API.

Dane wejściowe:
- pliki Markdown znajdujące się w katalogu `data/programy/`.

Użycie:
    python scripts/build_faiss_index_demo.py
    python scripts/build_faiss_index_demo.py --rebuild
"""

import argparse
from pathlib import Path

import pyarrow.parquet as pq
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _ROOT / "data" / "programy"
INDEX_DIR = _ROOT / "data" / "indexes" / "faiss_demo"
PARQUET_PATH = _ROOT / "data" / "szkolenia.parquet"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _load_metadata_by_file() -> dict[str, dict]:
    """Mapuje nazwę pliku markdown ('plik' w parquecie) na strukturalne dane
    szkolenia (nazwa, kategoria, dni, pdf_url) — ten sam schemat, co indeks Chroma."""
    rows = pq.read_table(str(PARQUET_PATH)).to_pylist()
    return {row["plik"]: row for row in rows}


def _load_documents():
    loader = DirectoryLoader(
        str(DATA_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()
    print(f"Wczytano {len(docs)} plików z {DATA_DIR}")

    meta_by_file = _load_metadata_by_file()
    matched = 0
    for doc in docs:
        filename = Path(doc.metadata["source"]).name
        row = meta_by_file.get(filename)
        if row:
            doc.metadata["nazwa"] = row["nazwa"]
            doc.metadata["kategoria"] = row["kategoria"]
            doc.metadata["dni"] = row["dni"]
            doc.metadata["pdf_url"] = row["pdf_url"]
            matched += 1
    print(
        f"Dopasowano metadane (nazwa/kategoria/pdf_url) dla {matched}/{len(docs)} plików"
    )
    return docs


def build_index(rebuild: bool = False) -> None:
    if INDEX_DIR.is_dir() and any(INDEX_DIR.iterdir()) and not rebuild:
        print(f"Indeks już istnieje: {INDEX_DIR}")
        print("Uruchom z parametrem --rebuild, aby przebudować indeks.")
        return

    docs = _load_documents()
    if not docs:
        print(f"Nie znaleziono plików Markdown w {DATA_DIR}")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
    )
    doc_splits = text_splitter.split_documents(docs)
    print(f"Utworzono {len(doc_splits)} fragmentów dokumentów")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(doc_splits, embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(INDEX_DIR))

    print(f"Zapisano indeks FAISS w {INDEX_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Przebuduj indeks od początku.",
    )
    args = parser.parse_args()
    build_index(rebuild=args.rebuild)


if __name__ == "__main__":
    main()
