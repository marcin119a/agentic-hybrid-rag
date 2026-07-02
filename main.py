import uuid

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError

from src.core.graph import graph

RECURSION_LIMIT = 5


def main():
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": RECURSION_LIMIT}
    print("Asystent Sages. Wpisz 'exit' albo 'quit', żeby zakończyć.\n")

    while True:
        question = input("Ty: ").strip()
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break

        try:
            result = graph.invoke({"messages": [HumanMessage(content=question)]}, config=config)
        except GraphRecursionError:
            print("Bot: Nie udało mi się dopasować szkolenia do tego pytania — spróbuj je inaczej sformułować.\n")
            continue

        answer = result["messages"][-1].content
        print(f"Bot: {answer}\n")


if __name__ == "__main__":
    main()