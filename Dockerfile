FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN apt-get update \
    && apt-get install -y --no-install-recommends unixodbc unixodbc-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt

COPY openssl.cnf /etc/ssl/openssl.cnf
COPY app app

EXPOSE 5000
ENV FLASK_APP=app.main

CMD ["python", "-m", "app.main"]
