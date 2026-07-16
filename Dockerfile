FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

COPY knowledge ./knowledge
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/.data \
    && chown -R appuser:appuser /app

USER appuser
VOLUME ["/app/.data"]
EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "chatbot.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

