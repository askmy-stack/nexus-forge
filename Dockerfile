FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./
COPY src/ src/
COPY config/ config/

RUN pip install uv && uv sync --no-dev

EXPOSE 8080

CMD ["uv", "run", "uvicorn", "textSummarizer.serving.app:app", "--host", "0.0.0.0", "--port", "8080"]
