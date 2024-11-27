from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import os

app = Flask(__name__)

# Función para hacer la predicción
def prediccion_accion(ticker):
    # Descarga de datos del mercado con yfinance
    datos = yf.download(ticker).reset_index()
    datos = datos[(datos['Date'] >= "2014-01-01") & (datos['Date'] <= "2023-12-12")].reset_index(drop=True)
    
    # Escalado de los precios de cierre
    scaler = MinMaxScaler(feature_range=(0,1))
    datos['valores_escalados'] = scaler.fit_transform(datos['Close'].values.reshape(-1,1))
    
    # División de los datos en conjuntos de entrenamiento y prueba
    datos_entrenamiento = datos[datos['Date'] < '2023-01-01']
    datos_prueba = datos[datos['Date'] >= '2023-01-01']
    
    # Preparación de los datos de entrenamiento
    x_entrenamiento = []
    y_entrenamiento = []
    
    for i in range(60, len(datos_entrenamiento['valores_escalados'])):
        x_entrenamiento.append(datos_entrenamiento['valores_escalados'][i-60:i])
        y_entrenamiento.append(datos_entrenamiento['valores_escalados'][i])
    
    x_entrenamiento = np.array(x_entrenamiento)
    y_entrenamiento = np.array(y_entrenamiento)
    x_entrenamiento = np.reshape(x_entrenamiento, (x_entrenamiento.shape[0], x_entrenamiento.shape[1], 1))
    
    x_prueba = []
    y_prueba = datos_prueba['valores_escalados']
    
    for i in range(60, len(datos_prueba)):
        x_prueba.append(datos_prueba['valores_escalados'][i-60:i])
    
    x_prueba = np.array(x_prueba)
    x_prueba = np.reshape(x_prueba, (x_prueba.shape[0], x_prueba.shape[1], 1))
    
    # Definición de la arquitectura del modelo LSTM
    modelo = Sequential()
    modelo.add(LSTM(units = 50, return_sequences = True, input_shape = (x_entrenamiento.shape[1], 1)))
    modelo.add(Dropout(0.25))
    modelo.add(LSTM(units = 50, return_sequences = True))
    modelo.add(Dropout(0.25))
    modelo.add(LSTM(units = 50, return_sequences = True))
    modelo.add(Dropout(0.25))
    modelo.add(LSTM(units = 50))
    modelo.add(Dropout(0.25))
    modelo.add(Dense(units = 1))
    
    modelo.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    # Entrenamiento del modelo
    modelo.fit(x_entrenamiento, y_entrenamiento, epochs = 10, batch_size = 32)
    
    # Predicción
    prediccion_precio_accion = modelo.predict(x_prueba)
    prediccion_precio_accion = scaler.inverse_transform(prediccion_precio_accion)
    
    # Trazado de la serie
    directorio_destino = 'static/img'
    if not os.path.exists(directorio_destino):
        os.makedirs(directorio_destino)
    
    plt.figure(figsize=(10,5), dpi=100)
    plt.plot(datos_entrenamiento['Date'], datos_entrenamiento['Close'], label='Datos de entrenamiento')
    plt.plot(datos_prueba['Date'], datos_prueba['Close'], color='blue', label='Precio actual del activo')
    plt.plot(datos_prueba[60:]['Date'], prediccion_precio_accion, color='orange', label='Precio pronosticado del activo')
    
    plt.title(f'Predicción del precio de {ticker}')
    plt.xlabel('Tiempo')
    plt.ylabel('Precio del activo')
    plt.legend(loc='upper left', fontsize=8)
    
    plt.savefig(os.path.join(directorio_destino, 'prediction_plot.png'))

    return os.path.join(directorio_destino, 'prediction_plot.png')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker']
        if ticker:
            image_path = prediccion_accion(ticker)
            return render_template('index.html', image_path=image_path, ticker=ticker)
    return render_template('index.html', image_path=None, ticker=None)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
