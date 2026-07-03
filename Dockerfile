FROM python:3.11-slim 

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
   curl \ 
   gcc \
   && rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir --upgrade pip

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"


COPY requirements.txt requirements.txt

RUN uv pip install --system --no-cache-dir -r requirements.txt

COPY pyproject.toml pyproject.toml


COPY src/ src/

RUN uv pip install --system -e . --no-deps

COPY data/indexes/chroma data/indexes/chroma 
COPY data/indexes/faiss_demo data/indexes/faiss_demo

COPY data/programy data/programy

COPY api.py api.py

COPY config.py config.py

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]