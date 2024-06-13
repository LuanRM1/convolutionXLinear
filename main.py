from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

app = Flask(__name__)
CORS(app)
modelo = load_model("modelo_mnist.h5")
modelo_linear = load_model("linear_model_mnist.h5")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    return predict_model(modelo)


@app.route("/predict_linear", methods=["POST"])
def predict_linear():
    return predict_model(modelo_linear)


def predict_model(model):
    file = request.files["file"]
    if not file:
        return "Nenhum arquivo enviado!", 400

    filestr = file.read()
    npimg = np.fromstring(filestr, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    start_time = time.time()  # inicio da medição do tempo
    prediction = model.predict(img)
    end_time = time.time()  # fim da medição do tempo

    inference_time = end_time - start_time  # Tempo de inferência
    predicted_class = np.argmax(prediction, axis=1)[0]

    return jsonify(
        {"prediction": int(predicted_class), "inference_time": inference_time}
    )


if __name__ == "__main__":
    app.run(debug=True, port=8000)
