from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from config import settings

response_model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0,
    api_key=settings.openai_api_key,
)
grader_model = ChatOpenAI(
    model="~anthropic/claude-haiku-latest",
    temperature=0.0,
    base_url=settings.openrouter_base_url,
    api_key=settings.openrouter_api_key,
)
