# Base: Debian 12 (bookworm)
FROM python:3.11-slim

# Ajustes generales
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/mssql-tools18/bin:${PATH}"

WORKDIR /app

# Asegura que fallen los pipes si algún comando falla
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# -------- Sistema + repos Microsoft (sin apt-key) --------
# - Instala curl/gnupg/ca-certificates y ODBC.
# - Agrega repo MS para Debian 12 con signed-by.
# - Instala msodbcsql18 y mssql-tools18.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg apt-transport-https \
        unixodbc unixodbc-dev; \
    mkdir -p /etc/apt/keyrings; \
    curl -fsSL https://packages.microsoft.com/keys/microsoft.asc \
      | gpg --dearmor -o /etc/apt/keyrings/microsoft.gpg; \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/microsoft.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" \
      > /etc/apt/sources.list.d/microsoft-prod.list; \
    apt-get update; \
    ACCEPT_EULA=Y apt-get install -y --no-install-recommends \
      msodbcsql18 mssql-tools18; \
    rm -rf /var/lib/apt/lists/*

# Copia e instala dependencias Python primero para aprovechar caché
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Config OpenSSL (si tu app lo requiere)
COPY openssl.cnf /etc/ssl/openssl.cnf

# Código de la aplicación
COPY app/ /app/app/

# Exposición y variables de ejecución
EXPOSE 5000
ENV FLASK_APP=app.main

CMD ["python", "-m", "app.main"]
