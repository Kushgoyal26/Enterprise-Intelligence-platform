# Production Dockerfile for the backend API.
#
# This is included as a deployment reference — it shows how the app would
# be containerized for production, even if you're running it directly with
# `uvicorn` during local development.
#
# Build:  docker build -t enterprise-intel-api .
# Run:    docker run -p 8000:8000 --env-file .env enterprise-intel-api

FROM python:3.11-slim

WORKDIR /app

# system deps needed by psycopg2 and some ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY . .

EXPOSE 8000

# a basic healthcheck so an orchestrator (Docker Compose, Kubernetes, etc.)
# can tell if the API is actually serving traffic
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
