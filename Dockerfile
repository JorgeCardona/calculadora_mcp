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

# 1. Copiar primero requirements.txt para aprovechar la caché de Docker
COPY requirements.txt /app/requirements.txt

# 2. Instalar dependencias desde el archivo requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copiar el resto del código de la aplicación
COPY . /app

# Exponer el puerto
EXPOSE 8000

# Comando de arranque
CMD ["python", "main.py"]
