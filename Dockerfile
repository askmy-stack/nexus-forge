FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
COPY src/ src/
COPY config/ config/
COPY params.yaml ./

RUN uv pip install --system -e .

EXPOSE 8080

CMD ["uvicorn", "textSummarizer.serving.app:app", "--host", "0.0.0.0", "--port", "8080"]
