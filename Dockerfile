# ==============================================================================
# NEEV: AI Building Facade Defect Screening Platform
# Production Multi-Stage Dockerfile
# ==============================================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    ENVIRONMENT=production

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements (CPU-only PyTorch keeps image under 512MB RAM)
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy application directories
COPY neev/ neev/
COPY prompts/ prompts/
COPY third_party/ third_party/
COPY runs/ runs/
COPY photo_Sample/ photo_Sample/
COPY .env.example .env

EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start production server (reads HOST and PORT dynamically from environment)
CMD ["python", "-m", "neev.app.main"]

