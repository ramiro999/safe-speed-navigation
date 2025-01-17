# Usar la imagen oficial de Python 3.11 slim como base
FROM python:3.11-slim

# Variables de entorno
ENV PYTHONUNBUFFERED=1

# Actualizar sistema e instalar herramientas esenciales
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    gcc \
    g++ \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Descargar e instalar manualmente GCC más reciente
RUN wget https://ftp.gnu.org/gnu/gcc/gcc-13.1.0/gcc-13.1.0.tar.gz \
    && tar -xzf gcc-13.1.0.tar.gz \
    && cd gcc-13.1.0 \
    && ./contrib/download_prerequisites \
    && mkdir build && cd build \
    && ../configure --enable-languages=c,c++ --disable-multilib \
    && make -j$(nproc) \
    && make install \
    && cd ../.. \
    && rm -rf gcc-13.1.0 gcc-13.1.0.tar.gz

# Actualizar enlaces simbólicos para libstdc++
RUN ln -sf /usr/local/lib64/libstdc++.so.6 /usr/lib/x86_64-linux-gnu/libstdc++.so.6

# Configurar el directorio de trabajo
WORKDIR /safe-speed-navigation

# Copiar archivo de dependencias
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de los archivos del proyecto
COPY . .

# Navegar al directorio adecuado y ejecutar el script make.sh
RUN cd stereo/NMRF/ops && sh make.sh && cd ../../..

# Ejecutar la aplicación
CMD ["python", "app.py", "0.0.0.0:8000"]