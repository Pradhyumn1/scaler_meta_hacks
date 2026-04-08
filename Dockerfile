FROM python:3.10-slim

# Create a non-root user as per Hugging Face guidelines
RUN useradd -m -u 1000 user

# Switch to the new user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy and install dependencies securely as the new user
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all remaining source code inside
COPY --chown=user . /app

EXPOSE 7860

CMD ["python", "-m", "server.app"]
