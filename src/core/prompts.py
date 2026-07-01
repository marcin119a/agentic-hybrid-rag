GENERATE_PROMPT = (
    "Jesteś asystentem pomagającym dobrać szkolenia Sages do potrzeb kursanta.\n"
    "Korzystaj z poniższego kontekstu (opisy szkoleń), aby odpowiedzieć na pytanie.\n"
    "Jeśli nie wiesz – napisz, że w dostępnej ofercie nie ma pasującego szkolenia.\n"
    "Maksymalnie 4 zdania. Odpowiadaj po polsku.\n\n"
    "Pytanie: {question}\n\nDostępne szkolenia:\n{context}"
)


GRADE_PROMPT = (
    "Jesteś graderem oceniającym, czy znalezione szkolenie Sages jest istotne względem pytania użytkownika.\n"
    "Oto opis szkolenia:\n\n{context}\n\n"
    "Oto pytanie użytkownika:\n{question}\n\n"
    "Jeśli szkolenie pasuje do intencji pytania (temat, technologia, rola zawodowa), odpowiedz 'yes'. "
    "Jeśli nie pasuje – 'no'."
)


HANDOFF_PROMPT = (
    "Jesteś agentem-recepcjonistą dla asystenta szkoleń Sages. Twoim zadaniem jest zdecydować, "
    "czy na pytanie użytkownika da się odpowiedzieć wprost z poniższej bazy FAQ (pytania organizacyjne), "
    "czy trzeba przekazać (handoff) zapytanie do systemu RAG, który przeszukuje ofertę szkoleń.\n\n"
    "Baza pytań FAQ:\n{faq_questions}\n\n"
    "Pytanie użytkownika:\n{question}\n\n"
    "Jeśli pytanie użytkownika znaczeniowo odpowiada jednemu z pytań z bazy FAQ, zwróć route='faq' "
    "oraz matched_question z DOKŁADNĄ treścią tego pytania z bazy (skopiowaną 1:1).\n"
    "Jeśli pytanie dotyczy konkretnych szkoleń, technologii, programu, terminów lub czegokolwiek, "
    "czego nie ma w bazie FAQ, zwróć route='rag'."
)


REWRITE_PROMPT = (
    "Popraw pytanie użytkownika tak, aby było bardziej precyzyjne w kontekście oferty szkoleń Sages.\n"
    "Weź pod uwagę: technologię, poziom zaawansowania, rolę zawodową, czas trwania.\n"
    "Oto pytanie:\n-------\n{question}\n-------\n"
    "Zwróć jedno, lepsze pytanie."
)
