# Usa una imagen base oficial de Python
FROM python:3.11.4-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo de requerimientos (requirements.txt) al contenedor
COPY requirements.txt .

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install flask-login


# Copia el resto de los archivos de la aplicación al contenedor
COPY . .

# Expone el puerto en el que la aplicación escuchará (opcional, depende de tu app)
EXPOSE 5000

# Define el comando de inicio para la aplicación
CMD ["python", "app.py"]
