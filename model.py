from model import prepare_data, prepare_lstm_data, create_lstm_model, make_prediction, calculate_errors

@app.route('/')
def index():
    # Preparar datos
    datos, datos_entrenamiento, datos_prueba, scaler = prepare_data()
    x_entrenamiento, y_entrenamiento, x_prueba, y_prueba = prepare_lstm_data(datos_entrenamiento, datos_prueba)

    # Crear y entrenar el modelo LSTM
    modelo = create_lstm_model(x_entrenamiento, y_entrenamiento)

    # Realizar predicción y guardar el gráfico
    prediccion_precio_accion, datos_y_verdaderos = make_prediction(modelo, x_prueba, datos_prueba, scaler)

    # Calcular errores
    mse, rmse, mape = calculate_errors(datos_y_verdaderos, prediccion_precio_accion)
    
    return render_template('index.html', mse=mse, rmse=rmse, mape=mape)
