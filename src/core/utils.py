from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage


def _current_question(state: MessagesState) -> str:
    """Ostatnie pytanie użytkownika (nie messages[0] — checkpointer trzyma całą historię sesji)."""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return message.content
    return state["messages"][0].content
