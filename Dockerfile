FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app app

EXPOSE 5000
ENV FLASK_APP=app.main

CMD ["python", "-m", "app.main"]
