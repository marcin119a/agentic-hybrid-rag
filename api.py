import uuid
from fastapi import FastAPI, HTTPException

from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel
from src.core.graph import graph

RECURSION_LIMIT = 15

app = FastAPI(title="ChatBot API")


class ChatRequest(BaseModel):
    question: str
    thread_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    thread_id: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }

    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=request.question)]}, config=config
        )
    except GraphRecursionError:
        return ChatResponse(
            answer="Nie udało mi się dopasować szkolenia do tego pytania — spróbuj je inaczej sformułować.",
            thread_id=thread_id,
        )

    answer = result["messages"][-1].content
    return ChatResponse(answer=answer, thread_id=thread_id)
