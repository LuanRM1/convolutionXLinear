import time
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from tensorflow.keras.models import load_model
import numpy as np

(x_treino, y_treino), (x_teste, y_teste) = mnist.load_data()

y_treino_cat = to_categorical(y_treino)
y_teste_cat = to_categorical(y_teste)

x_treino = x_treino / x_treino.max()
x_teste = x_teste / x_teste.max()

x_treino = x_treino.reshape(-1, 28, 28, 1)
x_teste = x_teste.reshape(-1, 28, 28, 1)

linear_model = Sequential(
    [
        Flatten(input_shape=(28, 28, 1)),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(10, activation="softmax"),
    ]
)

linear_model.compile(
    loss="categorical_crossentropy", metrics=["accuracy"], optimizer=Adam()
)

start_time = time.time()
historico = linear_model.fit(
    x_treino, y_treino_cat, epochs=10, validation_split=0.2, batch_size=32
)
end_time = time.time()

training_time = end_time - start_time
print(f"Tempo de treinamento: {training_time} segundos")

linear_model.save("linear_model_mnist.h5")

test_loss, test_acc = linear_model.evaluate(x_teste, y_teste_cat)
print(f"Acurácia: {test_acc}")

modelo_2 = load_model("linear_model_mnist.h5")

predicao = modelo_2.predict(x_teste[6].reshape(1, 28, 28, 1))
print(predicao)
print(np.argmax(predicao))
