"""Testy jednostkowe dla node'ów workflow (core/nodes.py, core/graph.py).

Fixture'y `make_fake_retriever_tool` i `make_state` — patrz conftest.py.
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

import core.nodes as nodes
from core.validation import TrainingAnswer, TrainingRecommendation


def test_retriever_tool_formats_documents_from_fake_retriever(make_fake_retriever_tool):
    """Sprawdza samo okablowanie: fake retriever -> create_retriever_tool -> sformatowany tekst."""
    docs = [
        Document(
            page_content="Szkolenie z Pythona dla początkujących.",
            metadata={"pdf_url": "http://example.com/python.pdf"},
        )
    ]
    tool = make_fake_retriever_tool(docs)

    result = tool.invoke({"query": "python"})

    assert "Szkolenie z Pythona dla początkujących." in result
    assert "http://example.com/python.pdf" in result


def test_retrieve_faiss_uses_fake_search_tool(monkeypatch, make_state):
    fake_context = (
        "Szkolenie: Python\nOpis...\nLink do programu szkolenia: http://example.com"
    )
    fake_search_faiss = MagicMock()
    fake_search_faiss.invoke.return_value = fake_context
    monkeypatch.setattr(nodes, "search_faiss", fake_search_faiss)
    monkeypatch.setattr(nodes, "_extract_keywords", lambda question: "python")

    state = make_state("Szukam szkolenia z Pythona")
    result = nodes.retrieve_faiss(state)

    fake_search_faiss.invoke.assert_called_once_with({"query": "python"})
    assert result["messages"] == [AIMessage(content=fake_context)]


@pytest.mark.parametrize(
    "score,expected",
    [("yes", "generate_answer"), ("no", "retrieve_faiss")],
)
def test_grade_documents_routing(monkeypatch, make_state, score, expected):
    monkeypatch.setattr(nodes, "_grade", lambda question, context: score)
    state = make_state("Szukam szkolenia z Pythona", context="jakiś kontekst")
    assert nodes.grade_documents(state) == expected


@pytest.mark.parametrize(
    "score,expected",
    [("yes", "generate_answer"), ("no", "rewrite_question")],
)
def test_grade_faiss_documents_routing(monkeypatch, make_state, score, expected):
    monkeypatch.setattr(nodes, "_grade", lambda question, context: score)
    state = make_state("Szukam szkolenia z Pythona", context="kontekst z FAISS")
    assert nodes.grade_faiss_documents(state) == expected


def test_rewrite_question_uses_fake_model(monkeypatch, make_state):
    fake_response_model = MagicMock()
    fake_response_model.invoke.return_value = AIMessage(
        content="Jakie szkolenie z Pythona na poziomie zaawansowanym polecacie?"
    )
    monkeypatch.setattr(nodes, "response_model", fake_response_model)

    state = make_state("python szkolenie")
    result = nodes.rewrite_question(state)

    assert result["messages"] == [
        HumanMessage(
            content="Jakie szkolenie z Pythona na poziomie zaawansowanym polecacie?"
        )
    ]


def test_generate_answer_with_fake_structured_model(monkeypatch, make_state):
    fake_structured = TrainingAnswer(
        answer="Polecam szkolenie z Pythona.",
        trainings=[TrainingRecommendation(title="Python", link="http://example.com")],
    )
    fake_response_model = MagicMock()
    fake_response_model.with_structured_output.return_value.invoke.return_value = (
        fake_structured
    )
    monkeypatch.setattr(nodes, "response_model", fake_response_model)

    state = make_state(
        "Szukam szkolenia z Pythona", context="kontekst: szkolenie Python"
    )
    result = nodes.generate_answer(state)
    message = result["messages"][0]

    assert "Polecam szkolenie z Pythona." in message.content
    assert message.additional_kwargs["trainings"] == [
        {"title": "Python", "link": "http://example.com"}
    ]
