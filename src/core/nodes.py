from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState

from core.models import grader_model, response_model
from core.prompts import GENERATE_PROMPT, REWRITE_PROMPT, GRADE_PROMPT
from core.validation import GradeDocuments
from core.tools import retriever_tool


def generate_query_or_respond(state: MessagesState):
    """LLM decyduje: wywołać tool search_sages_trainings (RAG) albo odpowiedzieć od razu."""
    response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])
    if response.tool_calls:
        print(f"  [workflow] → retrieve: {response.tool_calls[0]['args']}")
    else:
        print(f"  [workflow] → odpowiedź bezpośrednia: {response.content[:120]}")
    return {"messages": [response]}


def generate_answer(state: MessagesState):
    """Generuje finalną odpowiedź na podstawie dokumentów zwróconych przez retriever."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([HumanMessage(content=prompt)])
    print(f"  [workflow] → odpowiedź z RAG: {response.content[:120]}")
    return {"messages": [response]}


def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    """Sprawdza, czy zwrócone fragmenty dokumentacji są istotne."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = grader_model.with_structured_output(GradeDocuments).invoke(
        [{"role": "user", "content": prompt}]
    )
    score = response.binary_score.strip().lower()
    print(f"  [workflow] grader → {score}")
    return "generate_answer" if score == "yes" else "rewrite_question"


def rewrite_question(state: MessagesState):
    question = state["messages"][0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    print(f"  [workflow] rewrite → {response.content[:120]}")
    return {"messages": [HumanMessage(content=response.content)]}