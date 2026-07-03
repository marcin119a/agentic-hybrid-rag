"""Baza najczęściej zadawanych pytań (FAQ) — pytania organizacyjne, na które
agent handoff może odpowiedzieć bez odpytywania RAG-a."""

FAQ_DATABASE: dict[str, str] = {
    "Jak mogę zapisać się na szkolenie?": (
        "Zapisu dokonasz przez formularz na stronie wybranego szkolenia lub kontaktując się "
        "z naszym działem obsługi klienta pod adresem kontakt@sages.com.pl."
    ),
    "Czy szkolenia odbywają się stacjonarnie czy online?": (
        "Większość szkoleń Sages dostępna jest w obu formach — stacjonarnej (Warszawa, Kraków) "
        "oraz online na żywo z trenerem. Formę wybierasz podczas zapisu."
    ),
    "Jakie formy płatności są akceptowane?": (
        "Akceptujemy przelewy tradycyjne, płatności kartą oraz płatność w ratach dla wybranych szkoleń."
    ),
    "Czy wystawiacie faktury?": (
        "Tak, po każdym szkoleniu wystawiamy fakturę VAT, którą przesyłamy na adres e-mail podany "
        "przy zapisie."
    ),
    "Czy szkolenie kończy się certyfikatem?": (
        "Tak, każdy uczestnik otrzymuje imienny certyfikat ukończenia szkolenia Sages."
    ),
    "Jak wygląda rezygnacja ze szkolenia?": (
        "Rezygnację można zgłosić mailowo najpóźniej 5 dni roboczych przed startem szkolenia — "
        "otrzymasz wtedy pełny zwrot wpłaty."
    ),
    "Czy oferujecie szkolenia zamknięte dla firm?": (
        "Tak, przygotowujemy dedykowane szkolenia zamknięte dopasowane do potrzeb zespołu — "
        "skontaktuj się z nami, aby ustalić zakres i termin."
    ),
    "Czy można rozłożyć płatność na raty?": (
        "Tak, dla większości szkoleń dostępna jest opcja płatności w 2 lub 3 ratach — wybierz ją "
        "w formularzu zapisu."
    ),
    "Jak skontaktować się z obsługą klienta?": (
        "Nasz dział obsługi klienta jest dostępny pod adresem kontakt@sages.com.pl oraz "
        "telefonicznie od poniedziałku do piątku w godzinach 8:00–16:00."
    ),
    "Czy materiały szkoleniowe są dostępne po zakończeniu kursu?": (
        "Tak, materiały szkoleniowe pozostają dostępne w naszej platformie e-learningowej "
        "bezterminowo po zakończeniu szkolenia."
    ),
}
