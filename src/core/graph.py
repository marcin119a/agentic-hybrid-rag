from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from core.nodes import (
    generate_answer,
    generate_query_or_respond,
    grade_documents,
    handoff_agent,
    grade_faiss_documents,
    retrieve_faiss,
    rewrite_question,
)
from core.tools import retriever_tool

checkpointer = MemorySaver()

workflow = StateGraph(MessagesState)
workflow.add_node(handoff_agent)
workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node(retrieve_faiss)
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

workflow.add_edge(START, "handoff_agent")
workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {"tools": "retrieve", END: END},
)
workflow.add_conditional_edges("retrieve", grade_documents)
workflow.add_conditional_edges("retrieve_faiss", grade_faiss_documents)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

graph = workflow.compile(checkpointer=checkpointer)
