FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.3.4

WORKDIR /app

RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-root --no-interaction --no-ansi

COPY app ./app
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
