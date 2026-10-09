# Tracker image (rendezvous, relay, credits). It runs no model.
#   docker build -t myriad-tracker app/
#   docker run -d -p 8500:8500 -v myriad-data:/data myriad-tracker
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /srv
COPY pyproject.toml uv.lock README.md ./
COPY myriad ./myriad
RUN uv sync --frozen --no-dev && useradd --system --uid 10001 myriad && mkdir /data && chown myriad /data

USER myriad
VOLUME /data
EXPOSE 8500
CMD ["/srv/.venv/bin/myriad", "tracker", "--host", "0.0.0.0", "--port", "8500", "--db", "/data/tracker.sqlite"]
