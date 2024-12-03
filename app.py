from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import os
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

app = Flask(__name__)
app.secret_key = 'clave_secreta_segura'

# Configuración de Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Simulación de base de datos
users = {'admin': {'password': '1234'}}

class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id) if user_id in users else None

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username]['password'] == password:
            user = User(username)
            login_user(user)
            flash('Inicio de sesión exitoso', 'success')
            return redirect(url_for('index'))
        else:
            flash('Usuario o contraseña incorrectos', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Sesión cerrada exitosamente', 'info')
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    if request.method == 'POST':
        ticker = request.form['ticker']
        if ticker:
            image_path, metrics = prediccion_accion(ticker)
            return render_template('index.html', image_path=image_path, ticker=ticker, metrics=metrics)
    return render_template('index.html', image_path=None, ticker=None, metrics=None)

# Función para hacer la predicción
def prediccion_accion(ticker):
    datos = yf.download(ticker).reset_index()
    datos = datos[(datos['Date'] >= "2014-01-01") & (datos['Date'] <= "2023-12-12")].reset_index(drop=True)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    datos['valores_escalados'] = scaler.fit_transform(datos['Close'].values.reshape(-1, 1))
    
    datos_entrenamiento = datos[datos['Date'] < '2023-01-01']
    datos_prueba = datos[datos['Date'] >= '2023-01-01']
    
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
    
    modelo = Sequential()
    modelo.add(LSTM(units=50, return_sequences=True, input_shape=(x_entrenamiento.shape[1], 1)))
    modelo.add(Dropout(0.25))
    modelo.add(LSTM(units=50, return_sequences=True))
    modelo.add(Dropout(0.25))
    modelo.add(LSTM(units=50, return_sequences=True))
    modelo.add(Dropout(0.25))
    modelo.add(LSTM(units=50))
    modelo.add(Dropout(0.25))
    modelo.add(Dense(units=1))
    
    modelo.compile(optimizer='adam', loss='mean_squared_error')
    modelo.fit(x_entrenamiento, y_entrenamiento, epochs=10, batch_size=32)
    
    prediccion_precio_accion = modelo.predict(x_prueba)
    prediccion_precio_accion = scaler.inverse_transform(prediccion_precio_accion)
    
    # Métricas
    y_prueba_invertido = scaler.inverse_transform(y_prueba[60:].values.reshape(-1, 1))
    mae = mean_absolute_error(y_prueba_invertido, prediccion_precio_accion)
    mse = mean_squared_error(y_prueba_invertido, prediccion_precio_accion)
    rmse = np.sqrt(mse)

    # Trazado del gráfico
    directorio_destino = 'static/img'
    if not os.path.exists(directorio_destino):
        os.makedirs(directorio_destino)
    
    plt.figure(figsize=(10, 5), dpi=100)
    plt.plot(datos_entrenamiento['Date'], datos_entrenamiento['Close'], label='Datos de entrenamiento')
    plt.plot(datos_prueba['Date'], datos_prueba['Close'], color='blue', label='Precio actual del activo')
    plt.plot(datos_prueba[60:]['Date'], prediccion_precio_accion, color='orange', label='Precio pronosticado del activo')
    
    plt.title(f'Predicción del precio de {ticker}')
    plt.xlabel('Tiempo')
    plt.ylabel('Precio del activo')
    plt.legend(loc='upper left', fontsize=8)
    
    plt.savefig(os.path.join(directorio_destino, 'prediction_plot.png'))

    return os.path.join(directorio_destino, 'prediction_plot.png'), {'mae': mae, 'mse': mse, 'rmse': rmse}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)