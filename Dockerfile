FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || true

COPY . .

RUN python -m compileall dexworld

EXPOSE 8765

CMD ["python", "-m", "dexworld.cli", "serve", "--port", "8765"]
