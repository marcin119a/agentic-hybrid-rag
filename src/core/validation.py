from typing import Literal, Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Relevance score: 'yes' jeśli istotne, 'no' jeśli nie")


class RouteDecision(BaseModel):
    """Decyzja agenta handoff: odpowiedzieć z bazy FAQ czy przekazać zapytanie do RAG."""

    route: Literal["faq", "rag"] = Field(
        description="'faq' jeśli pytanie pokrywa się z bazą FAQ, 'rag' jeśli wymaga wyszukania w bazie szkoleń"
    )
    matched_question: Optional[str] = Field(
        default=None,
        description="Dokładna treść pytania z bazy FAQ, które odpowiada pytaniu użytkownika (tylko gdy route='faq')",
    )
