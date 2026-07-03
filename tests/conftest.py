import os
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import create_retriever_tool

DOCUMENT_PROMPT = PromptTemplate.from_template(
    "{page_content}\nLink do programu szkolenia: {pdf_url}"
)


class FakeRetriever(BaseRetriever):
    """Zwraca ustalone dokumenty zamiast pytać Chroma/FAISS."""

    docs: list[Document] = []

    def _get_relevant_documents(self, query: str) -> list[Document]:
        return self.docs


_chroma_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "indexes",
    "chroma",
)
if not os.path.isdir(_chroma_dir) or not os.listdir(_chroma_dir):
    import core.retriever
    import core.hybrid_retriever

    core.retriever.get_retriever = lambda: MagicMock(spec=BaseRetriever)
    core.hybrid_retriever.get_hybrid_retriever = lambda *args, **kwargs: FakeRetriever(
        docs=[]
    )


@pytest.fixture
def make_fake_retriever_tool():
    """Fabryka retriever_tool spiętego z FakeRetriever, do testów okablowania toolów."""

    def _make(docs: list[Document]):
        return create_retriever_tool(
            FakeRetriever(docs=docs),
            "search_sages_trainings",
            "fake retriever do testów",
            document_prompt=DOCUMENT_PROMPT,
        )

    return _make


@pytest.fixture
def make_state():
    """Fabryka MessagesState: pytanie użytkownika + opcjonalny kontekst (AIMessage)."""

    def _make(question: str, context: str | None = None):
        messages = [HumanMessage(content=question)]
        if context is not None:
            messages.append(AIMessage(content=context))
        return {"messages": messages}

    return _make
