FROM python:3.12-slim

WORKDIR /app

# Copy requirements.txt from the parent directory
COPY ../requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt
COPY ./ai_planner .
COPY ../.env /app/.env

EXPOSE 8000

CMD ["uvicorn", "ai_planner:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
