FROM python:3.12-slim

# Hugging Face runs as a non-root user (UID 1000)
RUN useradd -m -u 1000 user
USER user

# Set up environment
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY --chown=user . .

# Hugging Face Spaces expose port 7860 by default
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
