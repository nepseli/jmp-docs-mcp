# Deploy-later image. Not needed for local use -- see README.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    JMPDOCS_PATHS__DATA_DIR=/data

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential curl \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -e .

COPY config.yaml ./
COPY scripts/ ./scripts/
COPY eval/ ./eval/

# The index is mounted at /data rather than baked in, so the image stays small
# and a rebuild does not require re-crawling.
VOLUME ["/data", "/models"]

EXPOSE 8501 8000

CMD ["streamlit", "run", "src/jmpdocs/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
