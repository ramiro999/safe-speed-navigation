# Usar una imagen base moderna
FROM python:3.11-slim

# Crear un usuario sin privilegios para ejecutar la aplicación
RUN groupadd -r appgroup && useradd -r -g appgroup -m appuser

# Asegurarse de que el directorio de inicio sea accesible
RUN mkdir -p /home/appuser && chown -R appuser:appgroup /home/appuser

# Establecer directorios temporales para Matplotlib y Torch
ENV MPLCONFIGDIR=/tmp
ENV TORCH_HOME=/tmp/torch

# Instalar herramientas necesarias para compilar código nativo y bibliotecas requeridas
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Descargar e instalar una versión más reciente de libstdc++.so.6
RUN wget http://ftp.us.debian.org/debian/pool/main/g/gcc-12/libstdc++6_12.2.0-14_amd64.deb && \
    dpkg -i libstdc++6_12.2.0-14_amd64.deb && \
    rm libstdc++6_12.2.0-14_amd64.deb

# Verificar que GLIBCXX_3.4.32 está disponible
RUN strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX

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

# Exponer el puerto de la aplicación
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["python", "app.py"]
