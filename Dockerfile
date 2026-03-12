FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and application files
COPY src/ src/
COPY app.py .
COPY run_id.txt .

# Copy baked-in model artifacts (from CI training step)
COPY models/ models/

# Set PYTHONPATH so grid_risk package is importable
ENV PYTHONPATH=/app/src

EXPOSE 9696

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9696"]
