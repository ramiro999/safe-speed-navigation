# Usar una imagen base de Python 3.11
FROM python:3.11-slim

# Crear un usuario sin privilegios para ejecutar la aplicación
RUN groupadd -r appgroup && useradd -r -g appgroup -m appuser

# Asegurarse de que el directorio de inicio sea accesible
RUN mkdir -p /home/appuser && chown -R appuser:appgroup /home/appuser

# Establecer un directorio temporal para Matplotlib
ENV MPLCONFIGDIR=/tmp

# Redirigir el caché de Torch
ENV TORCH_HOME=/tmp/torch

# Instalar dependencias necesarias para compilar código nativo y bibliotecas de OpenGL
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos del proyecto al contenedor
COPY . .

# Actualizar pip e instalar dependencias del proyecto
RUN pip install --upgrade pip --root-user-action=ignore
RUN pip install --no-cache-dir -r requirements.txt --root-user-action=ignore

# Compilar MultiScaleDeformableAttention
RUN cd stereo/NMRF/ops && sh make.sh

# Cambiar al usuario sin privilegios
USER appuser

# Exponer el puerto de la aplicación (ajusta si es diferente)
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["python", "app.py"]
