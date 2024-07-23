FROM python:3.12-slim

WORKDIR /app

# Copy requirements.txt from the parent directory
COPY ../requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt
COPY ./ai_optimizer .
COPY ../.env /app/.env

EXPOSE 8002

CMD ["uvicorn", "ai_optimizer:app", "--host", "0.0.0.0", "--port", "8001", "--log-level", "info"]
