from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, MessagesState
from langgraph.types import Command

from core.models import grader_model, response_model
from core.prompts import (
    GENERATE_PROMPT,
    REWRITE_PROMPT,
    GRADE_PROMPT,
    HANDOFF_PROMPT,
    KEYWORD_PROMPT,
)
from core.validation import GradeDocuments, RouteDecision, TrainingAnswer
from core.tools import retriever_tool, search_faiss
from core.faq import FAQ_DATABASE
from core.utils import _current_question


def handoff_agent(
    state: MessagesState,
) -> Command[Literal["generate_query_or_respond", "__end__"]]:
    """Agent-recepcjonista: decyduje, czy odpowiedzieć od razu z bazy FAQ,
    czy przekazać (handoff) zapytanie dalej do przepływu RAG."""
    question = _current_question(state)
    faq_questions = "\n".join(f"- {q}" for q in FAQ_DATABASE)
    prompt = HANDOFF_PROMPT.format(faq_questions=faq_questions, question=question)
    decision = grader_model.with_structured_output(RouteDecision).invoke(
        [{"role": "user", "content": prompt}]
    )

    if decision.route == "faq" and decision.matched_question in FAQ_DATABASE:
        answer = FAQ_DATABASE[decision.matched_question]
        print(f"  [handoff] → FAQ: '{decision.matched_question}'")
        return Command(goto=END, update={"messages": [AIMessage(content=answer)]})

    print("  [handoff] → RAG (generate_query_or_respond)")
    return Command(goto="generate_query_or_respond")


def generate_query_or_respond(state: MessagesState):
    """LLM decyduje: wywołać tool search_sages_trainings (RAG) albo odpowiedzieć od razu."""
    response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])
    if response.tool_calls:
        print(f"  [workflow] → retrieve: {response.tool_calls[0]['args']}")
    else:
        print(f"  [workflow] → odpowiedź bezpośrednia: {response.content[:20]}")
    return {"messages": [response]}


def generate_answer(state: MessagesState):
    question = _current_question(state)
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    structured = response_model.with_structured_output(TrainingAnswer).invoke(
        [{"role": "user", "content": prompt}]
    )
    content = structured.model_dump_json()
    print(f"  [workflow] odpowiedź: {content[:20]}")
    trainings = [t.model_dump() for t in structured.trainings]
    return {
        "messages": [
            AIMessage(content=content, additional_kwargs={"trainings": trainings})
        ]
    }


def _grade(question: str, context: str) -> str:
    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = grader_model.with_structured_output(GradeDocuments).invoke(
        [{"role": "user", "content": prompt}]
    )
    return response.binary_score.strip().lower()


def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "retrieve_faiss"]:
    """Sprawdza, czy zwrócone fragmenty dokumentacji są istotne."""
    question = _current_question(state)
    context = state["messages"][-1].content
    score = _grade(question, context)
    print(f"  [workflow] grader → {score}")
    return "generate_answer" if score == "yes" else "retrieve_faiss"


def _extract_keywords(question: str) -> str:
    """Słowa kluczowe (technologia/framework/temat) zamiast pełnego zdania — FAISS (BM25 + mały
    model dense) trafia lepiej w konkret niż w rozbudowane, przeformułowane pytanie."""
    prompt = KEYWORD_PROMPT.format(question=question)
    response = grader_model.invoke([{"role": "user", "content": prompt}])
    return response.content.strip()


def retrieve_faiss(state: MessagesState):
    """Handoff: Chroma nie znalazła trafnego kontekstu — spróbuj lokalnego indeksu FAISS (open-source, bez klucza OpenAI).
    Zapytanie ograniczone do słów kluczowych, bez rewrite'u całego pytania."""
    question = _current_question(state)
    keywords = _extract_keywords(question)
    print(
        f"  [workflow] → handoff: retrieve_faiss (lokalny indeks, HF embeddings) | słowa kluczowe: {keywords}"
    )
    result = search_faiss.invoke({"query": keywords})
    return {"messages": [AIMessage(content=result)]}


def grade_faiss_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    """Ocenia kontekst z FAISS. To już ostatnia deska ratunku przed przeformułowaniem pytania."""
    question = _current_question(state)
    context = state["messages"][-1].content
    score = _grade(question, context)
    print(f"  [workflow] grader (faiss) → {score}")
    return "generate_answer" if score == "yes" else "rewrite_question"


def rewrite_question(state: MessagesState):
    question = _current_question(state)
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    print(f"  [workflow] rewrite → {response.content[:120]}")
    return {"messages": [HumanMessage(content=response.content)]}
