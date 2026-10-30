FROM tblite-repair-mlflow:20260909
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.7.13
ENV UV_CACHE_DIR=/opt/uv-cache UV_LINK_MODE=copy
RUN uv venv /opt/verifier-bootstrap && \
    uv pip install --python /opt/verifier-bootstrap/bin/python pytest==8.4.1 && \
    uv pip install --python /opt/verifier-bootstrap/bin/python pandas==2.2.3 scikit-learn==1.7.2 requests==2.32.3 mlflow==2.19.0 && \
    uv pip freeze --python /opt/verifier-bootstrap/bin/python > /opt/verifier-bootstrap.lock
ENV UV_OFFLINE=1
LABEL evaluation.variant="breast-cancer-mlflow-offline-dependencies-v1"
