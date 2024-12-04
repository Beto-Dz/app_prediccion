import os

from signal import signal, SIGINT
from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from azure.storage.blob import BlobServiceClient
from io import BytesIO
import pandas as pd
from dotenv import load_dotenv



def handler(signal_received, frame):
    # SIGINT or  ctrl-C detected, exit without error
    exit(0)

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# pylint: disable=C0103
app = Flask(__name__)


def load_model_from_blob():
    try:
        connect_str = os.getenv('AZURE_STORAGE_KEY')
        container_name = 'contenedormodel'
        blob_name = 'random_forest_model.pkl'

        # Crear cliente para Azure Blob Storage
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        blob_client = blob_service_client.get_container_client(container_name).get_blob_client(blob_name)

        # Descargar el contenido del blob como datos binarios
        blob_data = blob_client.download_blob().readall()

        # Cargar el modelo desde el flujo de datos binario
        model = joblib.load(BytesIO(blob_data))
        return model

    except Exception as e:
        print(f"Error al cargar el modelo desde el blob: {e}")
        return None

# Función para cargar el scaler desde local
def load_scaler():
    try:
        scaler = joblib.load('scaler.pkl')
        return scaler
    except Exception as e:
        print(f"Error al cargar el escalador: {e}")
        return None
        
@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Cargar el modelo
        model = load_model_from_blob()
        if model is None:
            print('error modelo')
            return jsonify({'error': 'Error al cargar el modelo.'}), 500
            

        # Cargar el scaler
        scaler = load_scaler()
        if scaler is None:
            print('error scaler')
            return jsonify({'error': 'Error al cargar el scaler.'}), 500

        
        # Obtener datos del formulario
        data = request.form.to_dict(flat=True)
        #print(data)
        
        # Obtener datos del formulario
        dia_de_la_semana = int(data['dia_de_la_semana'])
        numero_de_avenida = int(data['numero_de_avenida'])
        colonia = int(data['colonia'])
        condiciones_naturales = int(data['condiciones_naturales'])
        restricciones_de_la_via = int(data['restricciones_de_la_via'])
        forma_de_accidentarse = int(data['forma_de_accidentarse'])
        contra_que_fue_el_impacto = int(data['contra_que_fue_el_impacto'])
        cantidad_de_vehiculos = int(data['cantidad_de_vehiculos'])
        danios_monto = int(data['danios_monto'])
        numero_de_muertos = int(data['numero_de_muertos'])
        mes = int(data['mes'])
        dia_del_ano = int(data['dia_del_ano'])

        # Crear un DataFrame con los mismos nombres de características que se usaron al entrenar el escalador
        feature_names = ['DIA DE LA SEMANA','NUMERO DE AVENIDA','COLONIA','CONDICIONES NATURALES','RESTRICCIONES DE LA VIA','FORMA DE ACCIDENTARSE','CONTRA QUE FUE EL IMPACTO','CANTIDAD DE VEHICULOS','DANIOS (MONTO APROXIMADO EN MXN)','NUMERO DE MUERTOS','MES','DIA_DEL_ANO']

        
        # Crear un array para el modelo
        input_data = [[
            dia_de_la_semana, numero_de_avenida, colonia, condiciones_naturales,
            restricciones_de_la_via, forma_de_accidentarse, contra_que_fue_el_impacto,
            cantidad_de_vehiculos, danios_monto, numero_de_muertos, mes, dia_del_ano
        ]]

        # crear un dataframe con los datos recibidos y las columnas indicadas
        input_data_df = pd.DataFrame(input_data, columns=feature_names)

        
        # Escalar los datos
        scaled_features = scaler.transform(input_data_df)

        # Realizar predicción
        prediction = model.predict(scaled_features)
        
        # Retornar la predicción en formato JSON
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        print(f"Error en /predict: {e}")  # Este mensaje aparecerá en los logs del servidor
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    signal(SIGINT, handler)
    server_port = os.environ.get('PORT', '8080')
    app.run(debug=False, port=server_port, host='0.0.0.0')
