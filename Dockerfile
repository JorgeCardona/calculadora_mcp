# Imagen base ligera con Python
FROM python:3.11-slim

# Evita que Python genere .pyc y fuerza logs inmediatos
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalar dependencias del sistema (opcional pero recomendado)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos del proyecto
COPY . /app

# Instalar dependencias Python
RUN pip install --no-cache-dir fastapi uvicorn pydantic fastmcp

# Exponer el puerto (FastMCP usa el que definas en PORT)
EXPOSE 8000

# Comando de arranque
CMD ["python", "main.py"] 
