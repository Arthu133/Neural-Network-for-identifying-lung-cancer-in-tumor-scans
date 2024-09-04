from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Aplicar CORS à aplicação Flask
CORS(app, resources={r"/*": {"origins": "*"}})

# Configurações para upload de arquivos
UPLOAD_FOLDER = 'backend/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def chestScanPrediction(path_file, _model):
    classes_dir = ["Adenocarcinoma", "Large cell carcinoma", "Normal", "Squamous cell carcinoma"]

    img = image.load_img(path_file, target_size=(350, 350))  # Ajuste o tamanho aqui

    # Normalizando a imagem
    norm_img = image.img_to_array(img) / 255.0

    # Convertendo a imagem para um array numpy
    input_arr_img = np.array([norm_img])

    # Obtendo as previsões
    pred = np.argmax(_model.predict(input_arr_img))

    # Imprimindo a previsão do modelo
    print("Predicted Label:", classes_dir[pred])

    return classes_dir[pred]

def limpar_pasta(pasta):
    for arquivo in os.listdir(pasta):
        caminho_arquivo = os.path.join(pasta, arquivo)
        if os.path.isfile(caminho_arquivo):
            os.remove(caminho_arquivo)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/detect-cancer', methods=['POST'])
def detect_cancer():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        model_eff = load_model("./backend/ct_effnet_best_model.hdf5")
        resultado = chestScanPrediction(os.path.join(app.config['UPLOAD_FOLDER'], filename), model_eff)
        limpar_pasta(app.config['UPLOAD_FOLDER'])

        response = jsonify({'type': resultado})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
