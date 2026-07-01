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


REWRITE_PROMPT = (
    "Popraw pytanie użytkownika tak, aby było bardziej precyzyjne w kontekście oferty szkoleń Sages.\n"
    "Weź pod uwagę: technologię, poziom zaawansowania, rolę zawodową, czas trwania.\n"
    "Oto pytanie:\n-------\n{question}\n-------\n"
    "Zwróć jedno, lepsze pytanie."
)
