from typing import Literal, Optional
from pydantic import BaseModel, Field


class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="Relevance score: 'yes' jeśli istotne, 'no' jeśli nie"
    )


class RouteDecision(BaseModel):
    """Decyzja agenta handoff: odpowiedzieć z bazy FAQ czy przekazać zapytanie do RAG."""

    route: Literal["faq", "rag"] = Field(
        description="'faq' jeśli pytanie pokrywa się z bazą FAQ, 'rag' jeśli wymaga wyszukania w bazie szkoleń"
    )
    matched_question: Optional[str] = Field(
        default=None,
        description="Dokładna treść pytania z bazy FAQ, które odpowiada pytaniu użytkownika (tylko gdy route='faq')",
    )


class TrainingRecommendation(BaseModel):
    title: str = Field(description="Nazwa polecanego szkolenia")
    link: str = Field(description="Link do programu szkolenia (pdf_url z kontekstu)")


class TrainingAnswer(BaseModel):
    answer: str = Field(
        description=(
            "Odpowiedź dla użytkownika po polsku, maksymalnie 4 zdania. "
            "Nie umieszczaj w niej adresów URL ani linków w formacie markdown — "
            "linki znajdują się wyłącznie w polu 'trainings'."
        )
    )
    trainings: list[TrainingRecommendation] = Field(
        default_factory=list,
        description="Lista pasujących szkoleń wraz z linkami do programów.",
    )
