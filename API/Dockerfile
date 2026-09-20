FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements-api.txt ./requirements-api.txt
RUN pip install --no-cache-dir -r requirements-api.txt

COPY API ./API
COPY src/__init__.py ./src/__init__.py
COPY src/etape2 ./src/etape2
COPY configs/config_etape2.yaml ./configs/config_etape2.yaml
COPY outputs/etape2/models ./outputs/etape2/models
COPY data ./data

EXPOSE 8000

CMD ["uvicorn", "API.api:app", "--host", "0.0.0.0", "--port", "8000"]