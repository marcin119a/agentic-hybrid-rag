"""Tworzy/aktualizuje dataset testowy w LangSmith na podstawie przykładów z run_eval.py.

Użycie:
    python scripts/build_langsmith_dataset.py
    python scripts/build_langsmith_dataset.py --name moj-dataset

Wymaga LANGSMITH_API_KEY (patrz hybrid_rag.config.settings / .env).
"""

import argparse

from langsmith import Client
import os
import sys

from run_eval import examples

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


DEFAULT_DATASET_NAME = "sages-rag-qa-v2"
DEFAULT_DATASET_DESCRIPTION = (
    "Zestaw testowy pytań o szkolenia Sages i oczekiwanych odpowiedzi "
    "(przeniesiony z lokalnej ewaluacji w scripts/run_eval.py)."
)


def build_dataset(
    name: str = DEFAULT_DATASET_NAME, description: str = DEFAULT_DATASET_DESCRIPTION
) -> None:
    client = Client()

    if client.has_dataset(dataset_name=name):
        dataset = client.read_dataset(dataset_name=name)
        print(
            f"Dataset '{name}' już istnieje (id={dataset.id}) — dodaję/aktualizuję przykłady."
        )
    else:
        dataset = client.create_dataset(dataset_name=name, description=description)
        print(f"Utworzono dataset '{name}' (id={dataset.id}).")

    inputs = [{"question": question} for question, _ in examples]
    outputs = [{"answer": expected} for _, expected in examples]

    client.create_examples(inputs=inputs, outputs=outputs, dataset_id=dataset.id)
    print(f"Dodano {len(examples)} przykładów do datasetu '{name}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name", default=DEFAULT_DATASET_NAME, help="Nazwa datasetu w LangSmith."
    )
    parser.add_argument(
        "--description", default=DEFAULT_DATASET_DESCRIPTION, help="Opis datasetu."
    )
    args = parser.parse_args()

    build_dataset(name=args.name, description=args.description)
