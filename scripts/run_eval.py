import os
import sys
import argparse
import uuid

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langsmith.schemas import Example, Run
from langsmith import evaluate

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.graph import graph
from config import settings

DEFAULT_DATASET_NAME = "sages-rag-qa-v2"

examples = [
    (
        "Jakie szkolenia z Pythona oferuje Sages?",
        "Sages oferuje kilkanaście kursów Python: od podstaw (Python podstawy), przez średnio zaawansowany i zaawansowany, po specjalistyczne — Machine Learning, Pandas, NumPy, Django, REST API, wizualizacja danych i inne.",
    ),
    (
        "Ile dni trwa szkolenie z budowy systemów RAG?",
        "Szkolenie 'Od API LLM do Agent-RAG – budowa systemów RAG' trwa 5 dni. Dostępne jest też krótsze 2-dniowe szkolenie 'Retrieval Augmented Generation (RAG) - systemy AI do wyszukiwania informacji'.",
    ),
    (
        "Czy Sages ma szkolenie z AI dla HR?",
        "Tak, Sages oferuje szkolenie 'AI dla HR – od rekrutacji po utrzymanie talentów' (2 dni), które uczy wykorzystania AI w rekrutacji, onboardingu i analizie zaangażowania pracowników zgodnie z RODO.",
    ),
    (
        "Jakie szkolenie z AI polecasz dla programisty DevOps?",
        "Dla programisty DevOps polecane jest 'AIOps w praktyce — AI jako narzędzie DevOps' (3 dni) lub 'AI DevOps z Claude, Cline i Terraform – CI/CD w praktyce' (2 dni).",
    ),
    (
        "Jak długo trwa szkolenie z Kubernetes?",
        "Sages oferuje kilka kursów Kubernetes: podstawy (KUBERNETES/BASICS), CKAD i CKA — czas trwania wynosi od 2 do 4 dni w zależności od poziomu.",
    ),
]

eval_model = ChatOpenAI(
    model="gpt-5.5",
    api_key=settings.openai_api_key,
)

EVAL_PROMPT = """Oceń, na ile odpowiedź asystenta jest poprawna względem oczekiwanej.

Pytanie użytkownika: {question}

Oczekiwana (wzorcowa) odpowiedź: {expected}

Odpowiedź asystenta: {actual}

Podaj score od 0.0 do 1.0 (1.0 = odpowiedź w pełni poprawna / pokrywa oczekiwaną, 0.0 = całkowicie błędna lub nie na temat)."""


class EvalScore(BaseModel):
    score: float = Field(description="Ocena 0.0–1.0")
    comment: str = Field(default="", description="Krótkie uzasadnienie oceny.")


def predict(question: str) -> str:
    result = graph.invoke({"messages": [HumanMessage(content=question)]})
    return result["messages"][-1].content if result.get("messages") else ""


def qa_correctness(question: str, expected: str, actual: str) -> dict:
    if not question or (not expected and not actual):
        return {"score": 0.0, "comment": ""}
    prompt = EVAL_PROMPT.format(question=question, expected=expected, actual=actual)
    result = eval_model.with_structured_output(EvalScore).invoke(
        [{"role": "user", "content": prompt}]
    )
    return {
        "score": max(0.0, min(1.0, float(result.score))),
        "comment": getattr(result, "comment", ""),
    }


def _extract_question(inputs: dict) -> str:
    """Wyciąga treść pytania z przykładu datasetu.

    Większość przykładów ma postać {"question": ...}, ale część mogła zostać
    dodana ręcznie w LangSmith z podglądu run'a i ma postać {"messages": [...]}.
    """
    if "question" in inputs:
        return inputs["question"]
    messages = inputs.get("messages") or []
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("type") == "human":
            return message.get("content", "")
    return ""


def qa_correctness_evaluator(run: Run, example: Example) -> dict:
    question = _extract_question(example.inputs)
    expected = example.outputs.get("answer", "") if example.outputs else ""
    actual = run.outputs.get("answer", "") if run.outputs else ""
    result = qa_correctness(question, expected, actual)
    return {
        "key": "qa_correctness",
        "score": result["score"],
        "comment": result["comment"],
    }


def target(inputs: dict) -> dict:
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    question = _extract_question(inputs)
    result = graph.invoke({"messages": [HumanMessage(content=question)]}, config=config)
    answer = result["messages"][-1].content if result.get("messages") else ""
    return {"answer": answer}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default=DEFAULT_DATASET_NAME, help="Nazwa datasetu w LangSmith."
    )
    parser.add_argument(
        "--prefix", default="sages-rag", help="Prefiks nazwy eksperymentu."
    )
    args = parser.parse_args()

    results = evaluate(
        target,
        data=args.dataset,
        evaluators=[qa_correctness_evaluator],
        experiment_prefix=args.prefix,
        description="Ewaluacja workflow Sages RAG (LLM-as-judge qa_correctness).",
    )

    print(
        f"\nGotowe — wyniki eksperymentu dostępne w LangSmith (dataset: {args.dataset})."
    )
