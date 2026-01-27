FROM python:3.10-slim

WORKDIR /app

# No system dependencies needed for pure PDF text extraction!

# Install Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py", "--mode", "full"]