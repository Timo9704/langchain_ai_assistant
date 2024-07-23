FROM python:3.12-slim

ENV http_proxy="http://specialinternetaccess-lb.telekom.de:8080"
ENV https_proxy="http://specialinternetaccess-lb.telekom.de:8080"
ENV HTTP_PROXY="http://specialinternetaccess-lb.telekom.de:8080"
ENV HTTPS_PROXY="http://specialinternetaccess-lb.telekom.de:8080"

WORKDIR /app

# Copy requirements.txt from the parent directory
COPY ../requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt
COPY ./ai_planner .
COPY ../.env /app/.env

EXPOSE 8001

CMD ["uvicorn", "ai_planner:app", "--host", "0.0.0.0", "--port", "8001", "--log-level", "info"]
