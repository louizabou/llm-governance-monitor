FROM python:3.10-slim

WORKDIR /app

RUN pip install fastapi uvicorn google-cloud-aiplatform ddtrace datadog-api-client

COPY app.py .

ENV PORT=8080

CMD ["ddtrace-run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]