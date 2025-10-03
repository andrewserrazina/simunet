FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1

WORKDIR /app
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# existing lines ...
COPY backend /app/backend

# NEW: copy frontend (static files) and tell the app where they are
COPY frontend /app/frontend
ENV SIMUNET_FRONTEND_DIR=/app/frontend


# Render sets PORT at runtime; default 8000 for local
ENV PORT=8000
EXPOSE 8000

WORKDIR /app/backend
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]

