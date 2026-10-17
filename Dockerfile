# ==========================================
# Base Image: lightweight production Python image.
# ==========================================

FROM python:3.12-slim

# ==========================================
# Set Working Directory: container root.
# ==========================================

WORKDIR /app

# ==========================================
# Install runtime dependencies only (pyproject.toml, no extras).
# Copied on its own first so this layer is cached between code changes.
# ==========================================

COPY pyproject.toml .

RUN pip install --no-cache-dir .

# ==========================================
# Copy Project Files: Copies repo into container.
# ==========================================

COPY . .

# ==========================================
# Make Start Script Executable: run FastAPI and Streamlit together
# ==========================================

RUN chmod +x start.sh

# ==========================================
# Expose Ports
# ==========================================

EXPOSE 8000
EXPOSE 8501

# ==========================================
# Start Application
# ==========================================

CMD ["./start.sh"]