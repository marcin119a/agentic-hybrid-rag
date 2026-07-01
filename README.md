
## Wymagania

- Python 3.11+
- Klucz API OpenRouter (modele czatu)
- Klucz API OpenAI (embeddingi `text-embedding-3-small`)

## Instalacja

Projekt używa [uv](https://docs.astral.sh/uv/) — szybkiego menedżera pakietów i środowisk wirtualnych dla Pythona.

Zainstaluj `uv` (jeśli jeszcze go nie masz):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows:
```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | more"
```

Utwórz środowisko wirtualne i zainstaluj zależności:

```bash
uv venv
uv pip install -r requirements.txt
```

Zainstaluj projekt w trybie edytowalnym, żeby pakiety `core` i `hybrid_rag` z `src/` były importowalne z dowolnego miejsca (bez ręcznego `sys.path.insert`):

```bash
uv pip install -e . --no-deps
```

Aktywuj środowisko:

```bash
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows
```

Skopiuj `.env.example` do `.env` i uzupełnij klucze:

```bash
cp .env.example .env
```