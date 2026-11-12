FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

   COPY honest_validation_v3.py .
   CMD ["python", "honest_validation_v3.py"]
